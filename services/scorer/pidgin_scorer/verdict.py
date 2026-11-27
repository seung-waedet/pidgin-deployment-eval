import json
import re

from pydantic import BaseModel, ConfigDict, ValidationError

from pidgin_dataset.schema import Item


class _Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CriterionVerdict(_Verdict):
    id: str
    reason: str
    met: bool


class AutoFailVerdict(_Verdict):
    index: int
    reason: str
    triggered: bool


class Verdict(_Verdict):
    understood_reason: str
    understood: bool
    criteria: list[CriterionVerdict]
    auto_fail: list[AutoFailVerdict]


class VerdictError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    keys = [key for key, _ in pairs]
    for key in keys:
        if keys.count(key) > 1:
            raise VerdictError(f"duplicate key in judge output: {key}")
    return dict(pairs)


_FENCE = re.compile(r"^```(?:json)?\s*\n(?P<body>.*)\n```$", re.DOTALL | re.IGNORECASE)


def parse_verdict(item: Item, text: str) -> Verdict:
    # The whole reply must be the verdict. Pulling an object out of surrounding prose
    # would also accept a verdict the judge was only quoting from the answer.
    body = text.strip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced["body"].strip()
    if not body.startswith("{"):
        raise VerdictError("judge output is not a single JSON object")
    try:
        raw = json.loads(body, object_pairs_hook=_reject_duplicate_keys)
        verdict = Verdict.model_validate(raw)
    except (json.JSONDecodeError, ValidationError) as error:
        raise VerdictError(f"unreadable judge output: {error}") from error

    criterion_ids = sorted(c.id for c in verdict.criteria)
    if criterion_ids != sorted(c.id for c in item.criteria):
        raise VerdictError(f"judge marked criteria {criterion_ids}, item has others")
    indices = sorted(a.index for a in verdict.auto_fail)
    if indices != list(range(len(item.auto_fail))):
        raise VerdictError(f"judge marked auto_fail {indices}, item has {len(item.auto_fail)}")
    return verdict
