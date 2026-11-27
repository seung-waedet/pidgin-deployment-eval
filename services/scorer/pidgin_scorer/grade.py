from collections import defaultdict
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from pidgin_dataset.schema import Expected, Item
from pidgin_scorer.numbers import contains_number
from pidgin_scorer.verdict import Verdict


class Outcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    JUDGE_ERROR = "judge_error"


class Grade(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    pair: str | None
    expected: Expected
    outcome: Outcome
    missing_facts: list[str] = []
    failed_critical: list[str] = []
    triggered_auto_fail: list[str] = []
    criteria: dict[str, bool] = {}
    understood: bool | None = None


def grade(item: Item, answer: str, verdict: Verdict | None) -> Grade:
    identity = {"item_id": item.id, "pair": item.pair, "expected": item.expected}
    missing = [fact.name for fact in item.facts if not contains_number(answer, fact.value)]
    if verdict is None:
        # A missing fact is a fail that code decided, so a judge outage cannot hide it.
        outcome = Outcome.FAIL if missing else Outcome.JUDGE_ERROR
        return Grade(**identity, outcome=outcome, missing_facts=missing)

    met = {c.id: c.met for c in verdict.criteria}
    failed = [c.id for c in item.criteria if c.critical and not met[c.id]]
    triggered = [item.auto_fail[a.index] for a in verdict.auto_fail if a.triggered]
    passed = not (missing or failed or triggered)
    return Grade(
        **identity,
        outcome=Outcome.PASS if passed else Outcome.FAIL,
        missing_facts=missing,
        failed_critical=failed,
        triggered_auto_fail=triggered,
        criteria=met,
        understood=verdict.understood,
    )


class Summary(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: int
    passed: int
    judge_errors: int
    answer_items: int
    answer_passed: int
    pairs: int
    pairs_passed: int
    flagged_without_credit: int


def summarize(grades: list[Grade]) -> Summary:
    graded = [g for g in grades if g.outcome is not Outcome.JUDGE_ERROR]
    answers = [g for g in graded if g.expected is Expected.ANSWER]

    twins: dict[str, dict[Expected, Grade]] = defaultdict(dict)
    for g in graded:
        if g.pair is not None:
            twins[g.pair][g.expected] = g
    complete = [t for t in twins.values() if len(t) == 2]

    def passed(g: Grade) -> bool:
        return g.outcome is Outcome.PASS

    return Summary(
        items=len(graded),
        passed=sum(map(passed, graded)),
        judge_errors=len(grades) - len(graded),
        answer_items=len(answers),
        answer_passed=sum(map(passed, answers)),
        pairs=len(complete),
        pairs_passed=sum(passed(t[Expected.ANSWER]) and passed(t[Expected.FLAG]) for t in complete),
        flagged_without_credit=sum(
            passed(t[Expected.FLAG]) and not passed(t[Expected.ANSWER]) for t in complete
        ),
    )
