import json
from typing import Any, Dict, List

import pytest
from haystack.dataclasses import ChatMessage

from dc_custom_component.components.preprocessors.validation_results_merger import (
    ValidationResultsMerger,
)


FIELD = "country_of_origin"

# Two patterns — "must be ISO-2" and "must not be on the sanctions list".
# Used for the correction re-check tests so we can exercise both pass and fail.
RULES: List[Dict[str, Any]] = [
    {
        "id": "country-shape",
        "version": 1,
        "name": "country-iso-2",
        "description": "Two uppercase letters.",
        "value": r"[A-Z]{2}",
        "value_type": "string",
        "validation_type": "boolean",
        "threshold": None,
    },
    {
        "id": "country-not-sanctioned",
        "version": 1,
        "name": "no-sanctioned-country",
        "description": "Must not be KP or IR.",
        "value": r"(?!(?:KP|IR)$)[A-Z]{2}",
        "value_type": "string",
        "validation_type": "boolean",
        "threshold": None,
    },
]


def _decode(out: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the JSON payload back out of the assistant ChatMessage."""
    assert "replies" in out and len(out["replies"]) == 1
    return cast_dict(json.loads(out["replies"][0].text))


def cast_dict(value: Any) -> Dict[str, Any]:
    assert isinstance(value, dict)
    return value


def test_regex_only_path_produces_field_response_without_llm_fields() -> None:
    merger = ValidationResultsMerger(field_id=FIELD, regex_rules=RULES)
    regex_results = [
        {"rule_id": "country-shape", "valid": True, "implementation_type": "regex"},
    ]

    payload = _decode(merger.run(regex_results=regex_results, llm_replies=None))

    assert payload == {
        FIELD: {
            "validation_results": regex_results,
            "field_correction": None,
            "changes_applied": None,
        }
    }


def test_semantic_results_default_implementation_type_to_prompt() -> None:
    merger = ValidationResultsMerger(field_id=FIELD, regex_rules=[])
    llm_payload = {
        FIELD: {
            "semantic_results": [
                {"rule_id": "sem-1", "valid": True},  # no implementation_type
                {"rule_id": "sem-2", "valid": False, "implementation_type": "custom"},
            ]
        }
    }
    reply = ChatMessage.from_assistant(json.dumps(llm_payload))

    payload = _decode(merger.run(regex_results=None, llm_replies=[reply]))

    semantic = payload[FIELD]["validation_results"]
    assert semantic[0]["implementation_type"] == "prompt"  # default added
    assert semantic[1]["implementation_type"] == "custom"  # preserved


def test_regex_and_semantic_results_are_concatenated_regex_first() -> None:
    merger = ValidationResultsMerger(field_id=FIELD, regex_rules=[])
    regex_results = [{"rule_id": "r1", "valid": True, "implementation_type": "regex"}]
    llm_payload = {FIELD: {"semantic_results": [{"rule_id": "s1", "valid": True}]}}
    reply = ChatMessage.from_assistant(json.dumps(llm_payload))

    payload = _decode(merger.run(regex_results=regex_results, llm_replies=[reply]))

    merged = payload[FIELD]["validation_results"]
    assert [row["rule_id"] for row in merged] == ["r1", "s1"]


def test_correction_passing_all_patterns_is_kept() -> None:
    merger = ValidationResultsMerger(field_id=FIELD, regex_rules=RULES)
    llm_payload = {
        FIELD: {
            "field_correction": "US",  # passes both shape and sanctions rules
            "changes_applied": "Corrected from invalid to US.",
        }
    }
    reply = ChatMessage.from_assistant(json.dumps(llm_payload))

    payload = _decode(merger.run(regex_results=None, llm_replies=[reply]))

    assert payload[FIELD]["field_correction"] == "US"
    assert payload[FIELD]["changes_applied"] == "Corrected from invalid to US."


def test_correction_failing_regex_recheck_is_nulled_with_note() -> None:
    merger = ValidationResultsMerger(field_id=FIELD, regex_rules=RULES)

    # Sub-case A: prior changes_applied exists — note is appended.
    reply_a = ChatMessage.from_assistant(
        json.dumps(
            {
                FIELD: {"field_correction": "KP", "changes_applied": "Swapped value."},
            }
        )
    )
    payload_a = _decode(merger.run(regex_results=None, llm_replies=[reply_a]))
    assert payload_a[FIELD]["field_correction"] is None
    assert payload_a[FIELD]["changes_applied"] == (
        "Swapped value. AI-suggested correction failed regex re-check; manual review required."
    )

    # Sub-case B: no prior changes_applied — note alone.
    reply_b = ChatMessage.from_assistant(
        json.dumps(
            {
                FIELD: {"field_correction": "kp"},  # lowercase, fails shape rule
            }
        )
    )
    payload_b = _decode(merger.run(regex_results=None, llm_replies=[reply_b]))
    assert payload_b[FIELD]["field_correction"] is None
    assert payload_b[FIELD]["changes_applied"] == (
        "AI-suggested correction failed regex re-check; manual review required."
    )


def test_llm_reply_unwraps_when_nested_under_field_id_or_at_top_level() -> None:
    merger = ValidationResultsMerger(field_id=FIELD, regex_rules=[])
    sem_row = {"rule_id": "s1", "valid": True, "implementation_type": "prompt"}

    nested = ChatMessage.from_assistant(
        json.dumps({FIELD: {"semantic_results": [sem_row]}})
    )
    flat = ChatMessage.from_assistant(json.dumps({"semantic_results": [sem_row]}))

    payload_nested = _decode(merger.run(regex_results=None, llm_replies=[nested]))
    payload_flat = _decode(merger.run(regex_results=None, llm_replies=[flat]))

    assert payload_nested[FIELD]["validation_results"] == [sem_row]
    assert payload_flat[FIELD]["validation_results"] == [sem_row]


def test_malformed_llm_json_is_swallowed_and_returns_minimal_response() -> None:
    merger = ValidationResultsMerger(field_id=FIELD, regex_rules=[])
    reply = ChatMessage.from_assistant("not-json{{{")

    payload = _decode(merger.run(regex_results=None, llm_replies=[reply]))

    assert payload == {
        FIELD: {
            "validation_results": [],
            "field_correction": None,
            "changes_applied": None,
        }
    }


def test_invalid_regex_raises_at_init() -> None:
    bad_rule = {**RULES[0], "value": "([unclosed"}
    with pytest.raises(ValueError, match="Invalid regex for rule country-shape"):
        ValidationResultsMerger(field_id=FIELD, regex_rules=[bad_rule])
