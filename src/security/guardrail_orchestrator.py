"""
GuardrailOrchestrator — runs the full input / output guardrail stacks in
the right order and short-circuits on the first block, cheapest check first.

INPUT  side (before retrieval/agent touches the query):
    1. Input Rule Guardrail   (regex + keyword)   — cheapest, no model call
    2. PII Scanner            (regex)              — mask PII in the query itself
    3. Semantic Guardrail     (embedding cosine)   — catches paraphrased attacks

OUTPUT side (after answer generation + validation + reflection):
    4. Output Guardrail       (secret/leak regex)
    5. PII Scanner            (regex, output side) — mask PII the model copied
                                                       from a source document
"""

from __future__ import annotations

from dataclasses import dataclass

from src.security.input_rule_guardrail import InputRuleGuardrail
from src.security.output_guardrail import OutputGuardrail
from src.security.pii_scanner import PIIScanner
from src.security.schemas import GuardrailFinding, GuardrailVerdict
from src.security.semantic_guardrail import SemanticGuardrail


@dataclass
class InputCheckResult:
    allowed: bool
    normalized_query: str
    findings: list[GuardrailFinding]
    reason: str = ""


@dataclass
class OutputCheckResult:
    allowed: bool
    sanitized_answer: str
    findings: list[GuardrailFinding]
    reason: str = ""


class GuardrailOrchestrator:
    def __init__(self) -> None:
        self.input_rule = InputRuleGuardrail()
        self.pii = PIIScanner()
        self.semantic = SemanticGuardrail()
        self.output_rule = OutputGuardrail()

    def check_input(self, query: str) -> InputCheckResult:
        findings: list[GuardrailFinding] = []

        v1 = self.input_rule.check(query)
        findings += v1.findings
        if v1.blocked:
            return InputCheckResult(False, query, findings, v1.reason)

        v2 = self.pii.scan(v1.normalized_query, side="input")
        findings += v2.findings
        working_text = v2.sanitized_text if v2.findings else v1.normalized_query
        if v2.blocked:
            return InputCheckResult(False, working_text, findings, v2.reason)

        v3 = self.semantic.check(working_text)
        findings += v3.findings
        if v3.blocked:
            return InputCheckResult(False, working_text, findings, v3.reason)

        return InputCheckResult(True, working_text, findings)

    def check_output(self, answer: str) -> OutputCheckResult:
        findings: list[GuardrailFinding] = []

        v1 = self.output_rule.check(answer)
        findings += v1.findings
        working_text = v1.sanitized_text if v1.findings else answer
        if v1.blocked:
            return OutputCheckResult(False, working_text, findings, v1.reason)

        v2 = self.pii.scan(working_text, side="output")
        findings += v2.findings
        working_text = v2.sanitized_text if v2.findings else working_text
        if v2.blocked:
            return OutputCheckResult(False, working_text, findings, v2.reason)

        return OutputCheckResult(True, working_text, findings)
