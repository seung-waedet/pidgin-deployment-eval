from collections import Counter, defaultdict
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Family(StrEnum):
    MEDICATION = "medication"
    BANK_ALERT = "bank_alert"
    PUBLIC_NOTICE = "public_notice"


class Expected(StrEnum):
    ANSWER = "answer"
    FLAG = "flag"


class Review(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Fact(_Strict):
    name: str
    value: Decimal


class Criterion(_Strict):
    id: str
    text: str
    critical: bool = False


class Item(_Strict):
    id: str
    family: Family
    expected: Expected
    review: Review
    source: str
    question: str
    reference: str
    criteria: list[Criterion] = Field(min_length=1)
    facts: list[Fact] = []
    auto_fail: list[str] = []
    pair: str | None = None
    reviewer_note: str = ""

    @model_validator(mode="after")
    def _trap_names_its_pair(self) -> "Item":
        if self.expected is Expected.FLAG and self.pair is None:
            raise ValueError(f"trap {self.id} must name its pair")
        return self


class DatasetError(ValueError):
    pass


def check_dataset(items: list[Item]) -> None:
    for item_id, count in Counter(item.id for item in items).items():
        if count > 1:
            raise DatasetError(f"duplicate id: {item_id}")

    pairs: dict[str, list[Expected]] = defaultdict(list)
    for item in items:
        if item.pair is not None:
            pairs[item.pair].append(item.expected)
    for pair, expected in pairs.items():
        if sorted(expected) != [Expected.ANSWER, Expected.FLAG]:
            raise DatasetError(f"pair {pair} needs one answer and one flag")


def load_items(directory: Path) -> list[Item]:
    rows = []
    for path in sorted(directory.glob("*.yaml")):
        rows.extend(yaml.safe_load(path.read_text(encoding="utf-8")) or [])
    return [Item.model_validate(row) for row in rows]
