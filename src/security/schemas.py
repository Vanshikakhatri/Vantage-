"""Shared result dataclasses for the guardrail stack."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GuardrailFinding:
    layer: str            # "input_rule" | "semantic" | "pii" | "output_rule" | "output_pii"
    rule_or_intent: str    # which keyword / pattern / reference-intent / secret-type matched
    span: str               # the offending text (or masked preview)
    severity: str = "block"  # "block" | "mask" | "flag"


@dataclass
class GuardrailVerdict:
    allowed: bool
    normalized_query: str = ""
    sanitized_text: str = ""             # text after PII/secret masking is applied
    findings: list[GuardrailFinding] = field(default_factory=list)
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return not self.allowed
