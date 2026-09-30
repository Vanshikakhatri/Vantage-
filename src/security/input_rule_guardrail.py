"""
Layer 1 — Input Rule-Based Guardrail (Regex + Keyword).

Cheapest, fastest, zero-model-call check. Runs FIRST, before any LLM or
embedding call, so a malicious/garbage query never even reaches the agent.
Catches: known jailbreak phrases, prompt-injection markers, obviously
oversized payloads, suspicious encoded blobs.
"""

from __future__ import annotations

import re

from src.core.config import ApplicationConfig
from src.core.logging import LoggerManager
from src.security.schemas import GuardrailFinding, GuardrailVerdict

logger = LoggerManager.get_logger()


class InputRuleGuardrail:
    """Deterministic keyword + regex screen for user input."""

    def __init__(self) -> None:
        cfg = ApplicationConfig().guardrails["input_guardrail"]
        self.block_keywords = [kw.lower() for kw in cfg.get("block_keywords", [])]
        self.block_patterns = [re.compile(p) for p in cfg.get("block_patterns", [])]
        self.max_len = cfg.get("max_query_length", 4000)

    def check(self, text: str) -> GuardrailVerdict:
        findings: list[GuardrailFinding] = []

        if len(text) > self.max_len:
            findings.append(GuardrailFinding("input_rule", "max_length_exceeded", text[:80]))
            return GuardrailVerdict(
                allowed=False, findings=findings,
                reason=f"Query exceeds max length of {self.max_len} characters.",
            )

        lowered = text.lower()
        for kw in self.block_keywords:
            if kw in lowered:
                findings.append(GuardrailFinding("input_rule", kw, kw))

        for pattern in self.block_patterns:
            match = pattern.search(text)
            if match:
                findings.append(GuardrailFinding("input_rule", pattern.pattern, match.group(0)[:80]))

        if findings:
            logger.warning(f"InputRuleGuardrail blocked query: {[f.rule_or_intent for f in findings]}")
            return GuardrailVerdict(
                allowed=False, findings=findings,
                reason="Blocked by input rule guardrail (keyword/pattern match).",
            )

        return GuardrailVerdict(allowed=True, normalized_query=text.strip())
