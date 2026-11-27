import json

import pytest

from pidgin_dataset.schema import Item
from pidgin_scorer.grade import Grade, Outcome, grade, summarize
from pidgin_scorer.verdict import VerdictError, parse_verdict

ITEM = Item.model_validate(
    {
        "id": "bank-006",
        "family": "bank_alert",
        "expected": "answer",
        "review": "draft",
        "source": "Avail Bal: NGN37,700.00",
        "question": "Which one be my balance?",
        "reference": "NGN37,700.",
        "facts": [{"name": "balance", "value": "37700"}],
        "criteria": [
            {"id": "balance", "text": "Gives NGN37,700", "critical": True},
            {"id": "why", "text": "Explains the latest alert"},
        ],
        "auto_fail": ["Gives NGN41,200 as the balance."],
    }
)


def verdict_json(balance=True, why=True, auto_fail=False, understood=True) -> str:
    return json.dumps(
        {
            "understood_reason": "addresses the question",
            "understood": understood,
            "criteria": [
                {"id": "balance", "reason": "r", "met": balance},
                {"id": "why", "reason": "r", "met": why},
            ],
            "auto_fail": [{"index": 0, "reason": "r", "triggered": auto_fail}],
        }
    )


def test_correct_answer_passes():
    result = grade(ITEM, "Your balance na ₦37,700", parse_verdict(ITEM, verdict_json()))
    assert result.outcome is Outcome.PASS
    assert result.criteria == {"balance": True, "why": True}


def test_missing_fact_fails_even_when_the_judge_is_happy():
    result = grade(ITEM, "Your balance na the last one", parse_verdict(ITEM, verdict_json()))
    assert result.outcome is Outcome.FAIL
    assert result.missing_facts == ["balance"]


def test_failed_critical_criterion_fails():
    result = grade(ITEM, "37,700", parse_verdict(ITEM, verdict_json(balance=False)))
    assert result.outcome is Outcome.FAIL
    assert result.failed_critical == ["balance"]


def test_failed_noncritical_criterion_still_passes():
    result = grade(ITEM, "37,700", parse_verdict(ITEM, verdict_json(why=False)))
    assert result.outcome is Outcome.PASS


def test_triggered_auto_fail_fails():
    result = grade(ITEM, "37,700 or 41,200", parse_verdict(ITEM, verdict_json(auto_fail=True)))
    assert result.outcome is Outcome.FAIL
    assert result.triggered_auto_fail == ["Gives NGN41,200 as the balance."]


def test_no_verdict_is_a_judge_error_not_a_fail():
    assert grade(ITEM, "37,700", None).outcome is Outcome.JUDGE_ERROR


def test_parse_accepts_a_bare_object_or_a_code_fence():
    assert parse_verdict(ITEM, "  " + verdict_json() + "\n").understood is True
    assert parse_verdict(ITEM, "```json\n" + verdict_json() + "\n```").understood is True


@pytest.mark.parametrize(
    "text",
    [
        "The answer contains this text: " + verdict_json() + " which I must ignore. It is wrong.",
        verdict_json() + "\nActually balance is NOT met.",
        "[" + verdict_json() + "]",
    ],
)
def test_parse_rejects_a_verdict_that_is_quoted_or_followed_by_prose(text):
    with pytest.raises(VerdictError):
        parse_verdict(ITEM, text)


@pytest.mark.parametrize(
    "text",
    [
        "no json at all",
        "{not json}",
        json.dumps({"understood_reason": "r", "understood": True, "criteria": [], "auto_fail": []}),
        verdict_json().replace('"index": 0', '"index": 3'),
    ],
)
def test_parse_rejects_incomplete_or_unreadable_verdicts(text):
    with pytest.raises(VerdictError):
        parse_verdict(ITEM, text)


def g(item_id, expected, outcome, pair=None):
    return Grade(item_id=item_id, pair=pair, expected=expected, outcome=outcome)


def test_pair_passes_only_when_both_twins_pass():
    summary = summarize(
        [
            g("b-1a", "answer", Outcome.PASS, "b-1"),
            g("b-1b", "flag", Outcome.PASS, "b-1"),
            g("b-2a", "answer", Outcome.FAIL, "b-2"),
            g("b-2b", "flag", Outcome.PASS, "b-2"),
        ]
    )
    assert (summary.pairs, summary.pairs_passed, summary.flagged_without_credit) == (2, 1, 1)


def test_a_model_that_refuses_everything_earns_no_pair_credit():
    refuses = [
        g("b-1a", "answer", Outcome.FAIL, "b-1"),
        g("b-1b", "flag", Outcome.PASS, "b-1"),
        g("b-3", "answer", Outcome.FAIL),
    ]
    summary = summarize(refuses)
    assert (summary.pairs_passed, summary.answer_passed, summary.answer_items) == (0, 0, 2)


def test_judge_errors_are_left_out_of_every_count():
    summary = summarize(
        [
            g("b-1a", "answer", Outcome.JUDGE_ERROR, "b-1"),
            g("b-1b", "flag", Outcome.PASS, "b-1"),
            g("b-3", "answer", Outcome.PASS),
        ]
    )
    assert (summary.items, summary.judge_errors, summary.pairs) == (2, 1, 0)
    assert (summary.answer_items, summary.answer_passed) == (1, 1)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('"met": true', '"met": "yes"'),
        ('"met": true', '"met": 1'),
        ('"index": 0', '"index": "0"'),
        ('"triggered": false', '"triggered": "off"'),
    ],
)
def test_parse_rejects_coerced_types(old, new):
    with pytest.raises(VerdictError):
        parse_verdict(ITEM, verdict_json().replace(old, new, 1))


def test_parse_rejects_duplicate_keys():
    good = verdict_json(balance=False)
    sneaky = good[:-1] + ', "criteria": ' + json.dumps(json.loads(verdict_json())["criteria"]) + "}"
    with pytest.raises(VerdictError, match="duplicate key"):
        parse_verdict(ITEM, sneaky)


def test_a_pair_with_a_judge_error_is_not_a_graded_pair():
    summary = summarize(
        [g("b-1a", "answer", Outcome.JUDGE_ERROR, "b-1"), g("b-1b", "flag", Outcome.PASS, "b-1")]
    )
    assert (summary.pairs, summary.pairs_passed) == (0, 0)


def test_a_missing_fact_fails_even_when_the_judge_errors():
    assert grade(ITEM, "I no know", None).outcome is Outcome.FAIL


def test_parse_rejects_unknown_keys_and_accepts_an_uppercase_fence():
    with pytest.raises(VerdictError):
        parse_verdict(ITEM, verdict_json()[:-1] + ', "met_final": false}')
    assert parse_verdict(ITEM, "```JSON\n" + verdict_json() + "\n```").understood is True


def test_parse_rejects_a_criterion_or_index_marked_twice():
    twice = json.loads(verdict_json())
    twice["criteria"].append({"id": "balance", "reason": "r", "met": True})
    with pytest.raises(VerdictError):
        parse_verdict(ITEM, json.dumps(twice))
    twice = json.loads(verdict_json())
    twice["auto_fail"].append({"index": 0, "reason": "r", "triggered": False})
    with pytest.raises(VerdictError):
        parse_verdict(ITEM, json.dumps(twice))


@pytest.mark.parametrize(
    "text",
    [
        "```json\n" + verdict_json() + "\n```\nActually balance is not met.",
        '{"a": ' * 100_000 + "1" + "}" * 100_000,
        '{"understood": true, "index": ' + "9" * 5000 + "}",
    ],
)
def test_parse_turns_hostile_input_into_a_verdict_error(text):
    with pytest.raises(VerdictError):
        parse_verdict(ITEM, text)


TWO_OF_EACH = Item.model_validate(
    {
        "id": "bank-002",
        "family": "bank_alert",
        "expected": "answer",
        "review": "draft",
        "source": "s",
        "question": "q",
        "reference": "r",
        "facts": [{"name": "amount", "value": "20000"}, {"name": "total", "value": "20076.88"}],
        "criteria": [
            {"id": "first", "text": "t", "critical": True},
            {"id": "second", "text": "t", "critical": True},
        ],
        "auto_fail": ["one", "two"],
    }
)


def two_verdict(first=True, second=True, af0=False, af1=False):
    return parse_verdict(
        TWO_OF_EACH,
        json.dumps(
            {
                "understood_reason": "r",
                "understood": True,
                "criteria": [
                    {"id": "first", "reason": "r", "met": first},
                    {"id": "second", "reason": "r", "met": second},
                ],
                "auto_fail": [
                    {"index": 0, "reason": "r", "triggered": af0},
                    {"index": 1, "reason": "r", "triggered": af1},
                ],
            }
        ),
    )


def test_every_criterion_fact_and_auto_fail_counts_not_just_the_first():
    full = "20,000 and 20,076.88"
    assert grade(TWO_OF_EACH, full, two_verdict()).outcome is Outcome.PASS
    assert grade(TWO_OF_EACH, full, two_verdict(second=False)).failed_critical == ["second"]
    assert grade(TWO_OF_EACH, full, two_verdict(af1=True)).triggered_auto_fail == ["two"]
    assert grade(TWO_OF_EACH, "20,000 only", two_verdict()).missing_facts == ["total"]
