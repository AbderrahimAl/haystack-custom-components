import pytest

from dc_custom_component.components.preprocessors.regex_validator import RegexValidator


EMAIL_RULE = {
    "id": "r1",
    "version": 1,
    "name": "email-format",
    "description": "Value must look like an email.",
    "value": r"[^@\s]+@[^@\s]+\.[^@\s]+",
    "value_type": "string",
    "validation_type": "boolean",
    "threshold": None,
}


def test_passing_rule_reports_match_and_no_failures() -> None:
    validator = RegexValidator(field_id="email", regex_rules=[EMAIL_RULE])

    out = validator.run({"email": "alice@example.com"})

    assert out["any_failed"] is False
    assert out["failed_regex_hints"] == []
    assert len(out["regex_results"]) == 1
    assert out["regex_results"][0]["valid"] is True
    assert out["regex_results"][0]["rule_id"] == "r1"


def test_failing_rule_emits_hint_and_marks_any_failed() -> None:
    validator = RegexValidator(field_id="email", regex_rules=[EMAIL_RULE])

    out = validator.run({"email": "not-an-email"})

    assert out["any_failed"] is True
    assert out["regex_results"][0]["valid"] is False
    assert out["failed_regex_hints"] == [
        {
            "rule_id": "r1",
            "pattern": EMAIL_RULE["value"],
            "description": EMAIL_RULE["description"],
        }
    ]


def test_missing_field_is_treated_as_empty_string() -> None:
    validator = RegexValidator(field_id="email", regex_rules=[EMAIL_RULE])

    # alert has no "email" key, and explicit None should behave identically.
    out_missing = validator.run({})
    out_none = validator.run({"email": None})

    assert out_missing["any_failed"] is True
    assert out_none["any_failed"] is True
    assert out_missing["regex_results"][0]["valid"] is False
    assert out_none["regex_results"][0]["valid"] is False


def test_invalid_regex_raises_at_init() -> None:
    bad_rule = {**EMAIL_RULE, "value": "([unclosed"}

    with pytest.raises(ValueError, match="Invalid regex for rule r1"):
        RegexValidator(field_id="email", regex_rules=[bad_rule])
