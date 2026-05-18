import json
import re
from typing import Any, Dict, List, Optional, cast

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage


@component
class ValidationResultsMerger:
    """
    Combines deterministic regex results with the LLM's semantic verdicts and
    correction proposal into a single FieldValidationResponse per field.

    Bound at pipeline-build time with the field's regex rules so it can re-check
    the LLM-authored field_correction against every regex pattern attached to
    the field, not only those that originally failed (so a "fix" cannot swap one
    violation for another).

    Sits between the JsonSchemaValidator (LLM output) and the AnswerBuilder:
        schema_validator.validated -> merger.llm_replies
        regex_validator.regex_results -> merger.regex_results
        merger.replies -> answer_builder.replies
    """

    _CORRECTION_FAILED_NOTE = (
        "AI-suggested correction failed regex re-check; manual review required."
    )

    # See RegexValidator for the same convention. DB rule rows carry
    # non-serialisable columns (datetimes, etc.); we keep only the fields the
    # component actually uses, so to_dict() round-trips cleanly.
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
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResultsMerger":
        return cast("ValidationResultsMerger", default_from_dict(cls, data))

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        regex_results: Optional[List[Dict[str, Any]]] = None,
        llm_replies: Optional[List[ChatMessage]] = None,
    ) -> Dict[str, Any]:
        # Both inputs are optional so the merger can serve all field combinations:
        # - regex_results unconnected for pure-semantic fields (no RegexValidator wired)
        # - llm_replies   unconnected for pure-regex fields    (no LLM branch wired)
        regex_results = regex_results or []
        llm_output = self._parse_llm_replies(llm_replies or [])

        semantic_results = self._normalize_semantic_results(
            llm_output.get("semantic_results", [])
        )

        field_correction = llm_output.get("field_correction")
        changes_applied = llm_output.get("changes_applied")

        if field_correction is not None and not self._satisfies_all_patterns(
            field_correction
        ):
            field_correction = None
            changes_applied = (
                f"{changes_applied} {self._CORRECTION_FAILED_NOTE}"
                if changes_applied
                else self._CORRECTION_FAILED_NOTE
            )

        field_response = {
            self.field_id: {
                "validation_results": list(regex_results) + semantic_results,
                "field_correction": field_correction,
                "changes_applied": changes_applied,
            }
        }

        merged_message = ChatMessage.from_assistant(json.dumps(field_response))
        return {"replies": [merged_message]}

    def _parse_llm_replies(self, replies: List[ChatMessage]) -> Dict[str, Any]:
        if not replies:
            return {}
        text = replies[-1].text or ""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if (
            isinstance(data, dict)
            and self.field_id in data
            and isinstance(data[self.field_id], dict)
        ):
            return cast(Dict[str, Any], data[self.field_id])
        return cast(Dict[str, Any], data) if isinstance(data, dict) else {}

    @staticmethod
    def _normalize_semantic_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in rows:
            normalized = dict(row)
            normalized.setdefault("implementation_type", "prompt")
            out.append(normalized)
        return out

    def _satisfies_all_patterns(self, value: Any) -> bool:
        candidate = "" if value is None else str(value)
        for _, compiled in self._compiled:
            if compiled.fullmatch(candidate) is None:
                return False
        return True
