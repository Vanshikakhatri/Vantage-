"""
PII Scanner — regex rule-pack, applied on BOTH input and output.

Used in two places in the pipeline:
  1. INPUT side  : scans the user's query before retrieval. If the user pastes
                   someone's Aadhaar/PAN/phone/email etc. into the chat box, it
                   gets masked before the query is embedded, logged, or sent to
                   any LLM (including local Ollama — belt & braces).
  2. OUTPUT side : scans the generated answer before it is shown to the user,
                   in case a retrieved chunk contained PII that the model
                   copied verbatim into the response. This is the DPDP-Act
                   relevant control — Vantage should never *emit* personal
                   data it wasn't explicitly asked to surface for the
                   requesting user.

Rule-pack targets DPDP Act 2023 "personal data" categories (Aadhaar, PAN,
Indian mobile numbers, email, DOB, IFSC, passport) plus generic globals
(credit card). Swappable for presidio-analyzer if installed (see
`use_presidio`) for statistical/NER-based detection on top of regex.
"""

from __future__ import annotations

import re

from src.core.config import ApplicationConfig
from src.core.logging import LoggerManager
from src.security.schemas import GuardrailFinding, GuardrailVerdict

logger = LoggerManager.get_logger()

try:
    from presidio_analyzer import AnalyzerEngine
    _PRESIDIO_AVAILABLE = True
except Exception:
    _PRESIDIO_AVAILABLE = False


class PIIScanner:
    def __init__(self, use_presidio: bool = False) -> None:
        cfg = ApplicationConfig().guardrails["pii_scanner"]
        self.rules = {name: re.compile(pattern) for name, pattern in cfg["rules"].items()}
        self.action = cfg.get("action", "mask")
        self.mask_char = cfg.get("mask_char", "\u2588")
        self.use_presidio = use_presidio and _PRESIDIO_AVAILABLE
        if self.use_presidio:
            self._presidio = AnalyzerEngine()

    def scan(self, text: str, side: str = "input") -> GuardrailVerdict:
        """side: 'input' or 'output' — only affects the log message/tagging."""
        findings: list[GuardrailFinding] = []
        sanitized = text

        for name, pattern in self.rules.items():
            for match in pattern.finditer(text):
                span = match.group(0)
                findings.append(
                    GuardrailFinding(
                        layer=f"pii_{side}", rule_or_intent=name,
                        span=self._preview(span), severity=self.action,
                    )
                )
                if self.action == "mask":
                    sanitized = sanitized.replace(span, self.mask_char * len(span))

        if self.use_presidio:
            for result in self._presidio.analyze(text=text, language="en"):
                span = text[result.start:result.end]
                findings.append(
                    GuardrailFinding(
                        layer=f"pii_{side}_presidio", rule_or_intent=result.entity_type,
                        span=self._preview(span), severity=self.action,
                    )
                )
                if self.action == "mask":
                    sanitized = sanitized.replace(span, self.mask_char * len(span))

        if findings:
            logger.info(f"PIIScanner ({side}) found {len(findings)} matches: "
                        f"{[f.rule_or_intent for f in findings]}")

        blocked = self.action == "block" and bool(findings)
        return GuardrailVerdict(
            allowed=not blocked,
            sanitized_text=sanitized,
            findings=findings,
            reason="PII detected and blocked." if blocked else "",
        )

    @staticmethod
    def _preview(span: str) -> str:
        if len(span) <= 4:
            return "*" * len(span)
        return span[:2] + "*" * (len(span) - 4) + span[-2:]
