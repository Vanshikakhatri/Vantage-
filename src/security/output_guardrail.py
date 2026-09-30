"""
Output Guardrail — secret / credential leakage scanner.

Runs on the FINAL generated answer, after validation/reflection have already
decided the answer is good enough to ship. This is a hard, separate check:
even a well-grounded, high-confidence answer must never leak an API key,
private key block, bearer token, JWT, etc. that happened to sit inside a
source document (e.g. a config file that got accidentally ingested).

Order in the pipeline: Answer Generation -> Validation -> Reflection ->
**Output Guardrail (this)** -> PII Scanner (output side) -> Audit Log -> UI.
"""

from __future__ import annotations

import re

from src.core.config import ApplicationConfig
from src.core.logging import LoggerManager
from src.security.schemas import GuardrailFinding, GuardrailVerdict

logger = LoggerManager.get_logger()


class OutputGuardrail:
    def __init__(self) -> None:
        cfg = ApplicationConfig().guardrails["output_guardrail"]
        self.patterns = {name: re.compile(p) for name, p in cfg["secret_patterns"].items()}
        self.action = cfg.get("action", "mask")

    def check(self, answer: str) -> GuardrailVerdict:
        findings: list[GuardrailFinding] = []
        sanitized = answer

        for name, pattern in self.patterns.items():
            for match in pattern.finditer(answer):
                span = match.group(0)
                findings.append(GuardrailFinding("output_rule", name, span[:12] + "…"))
                if self.action == "mask":
                    sanitized = sanitized.replace(span, "[REDACTED_SECRET]")

        if findings:
            logger.warning(f"OutputGuardrail caught leakage: {[f.rule_or_intent for f in findings]}")

        blocked = self.action == "block" and bool(findings)
        return GuardrailVerdict(
            allowed=not blocked,
            sanitized_text=sanitized,
            findings=findings,
            reason="Secret/credential leakage blocked." if blocked else "",
        )
