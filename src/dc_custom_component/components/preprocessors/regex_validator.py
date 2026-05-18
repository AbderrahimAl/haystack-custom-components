import json
import re
from typing import Any, Dict, List, cast

from haystack import component, default_from_dict, default_to_dict


@component
class RegexValidator:
    """
    Deterministic evaluator for the regex rules of a single alert field.

    Bound at pipeline-build time with the field's regex rules. At run time, given
    the alert dict, emits:
      - regex_results:        one ValidationResult-shaped dict per regex rule.
      - failed_regex_hints:   {rule_id, pattern, description} per failed rule,
                              consumed by the prompt builder to constrain the
                              LLM-authored correction.
      - any_failed:           convenience flag for downstream branching.

    Patterns are matched with re.fullmatch — the value as a whole must satisfy
    the pattern. Rule authors that want substring semantics should write the
    pattern accordingly (e.g. `.*foo.*`).
    """

    # Fields we keep on the rule rows. DB rows carry extra columns (datetime
    # timestamps, audit fields, etc.) that Haystack's strict serialisation
    # rejects and that we do not need at runtime.
    _RULE_KEYS = (
        "id",
        "version",
        "name",
        "description",
        "value",
        "value_type",
        "validation_type",
        "threshold",
    )

    def __init__(self, field_id: str, regex_rules: List[Dict[str, Any]]):
        self.field_id = field_id
        # Sanitise once, store, then compile. The sanitised list is what
        # Haystack serialises via to_dict() and ships to Deepset.
        self.regex_rules = [{k: r.get(k) for k in self._RULE_KEYS} for r in regex_rules]
        self._compiled: List[tuple] = []
        for rule in self.regex_rules:
            pattern = str(rule.get("value") or "")
            try:
                compiled = re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"Invalid regex for rule {rule.get('id')} on field "
                    f"'{field_id}': {exc}"
                ) from exc
            self._compiled.append((rule, compiled))

    def to_dict(self) -> Dict[str, Any]:
        return cast(
            Dict[str, Any],
            default_to_dict(self, field_id=self.field_id, regex_rules=self.regex_rules),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegexValidator":
        return cast("RegexValidator", default_from_dict(cls, data))

    @component.output_types(
        regex_results=List[Dict[str, Any]],
        failed_regex_hints=List[Dict[str, Any]],
        any_failed=bool,
    )
    def run(self, alert: Any) -> Dict[str, Any]:
        # Accept either a dict or a JSON string. Deepset's hosted UI feeds the
        # pipeline `query` input as a string, which would otherwise reach this
        # component unparsed; in batch pipelines an OutputAdapter upstream
        # already emits a dict, so both shapes occur in practice.
        if isinstance(alert, str):
            try:
                alert = json.loads(alert)
            except json.JSONDecodeError:
                alert = {}
        if not isinstance(alert, dict):
            alert = {}

        raw_value = alert.get(self.field_id, "")
        value = "" if raw_value is None else str(raw_value)

        regex_results: List[Dict[str, Any]] = []
        failed_regex_hints: List[Dict[str, Any]] = []
        any_failed = False

        for rule, compiled in self._compiled:
            matched = compiled.fullmatch(value) is not None

            regex_results.append(
                {
                    "rule_id": rule.get("id"),
                    "rule_version": rule.get("version"),
                    "implementation_type": "regex",
                    "validation_type": rule.get("validation_type", "boolean"),
                    "threshold": rule.get("threshold"),
                    "valid": matched,
                    "severity_score": None,
                    "confidence_score": 1.0,
                    "justification": self._justification(
                        rule, value, compiled.pattern, matched
                    ),
                    "correction": None,
                }
            )

            if not matched:
                any_failed = True
                failed_regex_hints.append(
                    {
                        "rule_id": rule.get("id"),
                        "pattern": compiled.pattern,
                        "description": rule.get("description", ""),
                    }
                )

        return {
            "regex_results": regex_results,
            "failed_regex_hints": failed_regex_hints,
            "any_failed": any_failed,
        }

    @staticmethod
    def _justification(
        rule: Dict[str, Any], value: str, pattern: str, matched: bool
    ) -> str:
        name = rule.get("name") or f"rule {rule.get('id')}"
        if matched:
            return f"Value matches required pattern '{pattern}' ({name})."
        return f"Value '{value}' does not match required pattern '{pattern}' ({name})."
