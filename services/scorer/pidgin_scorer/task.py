from pathlib import Path
from typing import Literal

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import Metric, SampleScore, Score, Target, metric, scorer
from inspect_ai.solver import Generate, TaskState, generate, solver
from pydantic import BaseModel, ConfigDict

from pidgin_dataset.schema import Item, Review, check_dataset, load_items
from pidgin_scorer.grade import Grade, Outcome, Summary, grade, summarize
from pidgin_scorer.judge import judge

ITEMS_DIR = Path(__file__).resolve().parents[3] / "items"
CASES_FILE = ITEMS_DIR / "judge" / "cases.yaml"
GRADER_ROLE = "grader"
DEFAULT_GRADER = "claude-code/fable"


class ReleaseError(ValueError):
    pass


def load_checked(items_dir: str, release: bool) -> dict[str, Item]:
    items = load_items(Path(items_dir))
    check_dataset(items)
    drafts = [i.id for i in items if i.review is Review.DRAFT]
    if release and drafts:
        raise ReleaseError(
            f"{len(drafts)} items are still draft ({', '.join(drafts)}). "
            "Approve them, or pass -T release=false for a development run."
        )
    return {i.id: i for i in items}


def to_sample(item: Item, dataset_size: int) -> Sample:
    # No system prompt: the model sees what a real person would paste and ask.
    return Sample(
        id=item.id,
        input=f"{item.source.strip()}\n\n{item.question.strip()}",
        target=item.reference,
        metadata={
            "pair": item.pair,
            "expected": item.expected.value,
            "review": item.review.value,
            "dataset_size": dataset_size,
        },
    )


NAN = float("nan")


def _complete(scores: list[SampleScore]) -> bool:
    # Inspect drops errored samples before metrics and accepts subset runs, so a
    # number is only reported when every sample in the dataset was graded.
    expected = scores[0].sample_metadata["dataset_size"] if scores else 0
    return bool(scores) and len({s.sample_id for s in scores}) == expected


def _summary(scores: list[SampleScore]) -> Summary | None:
    grades = [Grade.model_validate(s.score.metadata["grade"]) for s in scores]
    if not _complete(scores) or any(g.outcome is Outcome.JUDGE_ERROR for g in grades):
        return None
    return summarize(grades)


def _ratio(passed: int, total: int) -> float:
    return passed / total if total else NAN


@metric
def item_accuracy() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        s = _summary(scores)
        return _ratio(s.passed, s.items) if s else NAN

    return compute


@metric
def answer_accuracy() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        s = _summary(scores)
        return _ratio(s.answer_passed, s.answer_items) if s else NAN

    return compute


@metric
def pair_score() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        s = _summary(scores)
        return _ratio(s.pairs_passed, s.pairs) if s else NAN

    return compute


@metric
def judge_errors() -> Metric:
    def compute(scores: list[SampleScore]) -> int:
        return sum(s.score.metadata["grade"]["outcome"] == Outcome.JUDGE_ERROR for s in scores)

    return compute


async def _grade(item: Item, answer: str) -> tuple[Grade, dict]:
    grader = get_model(role=GRADER_ROLE, default=DEFAULT_GRADER)
    verdict, raw = await judge(item, answer, grader)
    result = grade(item, answer, verdict)
    details = {
        "judge_model": str(grader),
        "grade": result.model_dump(mode="json"),
        "verdict": verdict.model_dump(mode="json") if verdict else None,
        "judge_raw": raw,
    }
    return result, details


def _explain(result: Grade) -> str:
    reasons = (
        [f"missing fact: {f}" for f in result.missing_facts]
        + [f"failed critical: {c}" for c in result.failed_critical]
        + [f"auto fail: {a}" for a in result.triggered_auto_fail]
    )
    return f"{result.outcome}: " + ("; ".join(reasons) or "all checks met")


@scorer(metrics=[item_accuracy(), answer_accuracy(), pair_score(), judge_errors()])
def pidgin_scorer(items_dir: str):
    items = load_checked(items_dir, release=False)

    async def score(state: TaskState, target: Target) -> Score:
        if state.epoch != 1:
            raise ValueError("pair scoring needs one grade per item: run with --epochs 1")
        answer = state.output.completion
        result, details = await _grade(items[str(state.sample_id)], answer)
        return Score(
            value=1 if result.outcome is Outcome.PASS else 0,
            answer=answer,
            explanation=_explain(result),
            metadata=details,
        )

    return score


@task
def pidgin_eval(items_dir: str = str(ITEMS_DIR), release: bool = True) -> Task:
    items = load_checked(items_dir, release)
    return Task(
        dataset=MemoryDataset([to_sample(i, len(items)) for i in items.values()]),
        solver=generate(),
        scorer=pidgin_scorer(items_dir),
        epochs=1,
        metadata={"item_ids": sorted(items)},
    )


class CalibrationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    item: str
    answer: str
    expected: Literal["pass", "fail"]
    why: str
    review: Review


@solver
def fixed_answer():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.output = ModelOutput.from_content(model="fixed", content=state.metadata["answer"])
        return state

    return solve


@metric
def agreement() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        if not _complete(scores):
            return NAN
        judged = [s for s in scores if s.score.metadata["grade"]["outcome"] != Outcome.JUDGE_ERROR]
        return _ratio(sum(s.score.value == 1 for s in judged), len(judged))

    return compute


@metric
def calibration_judge_errors() -> Metric:
    def compute(scores: list[SampleScore]) -> int:
        return sum(s.score.metadata["grade"]["outcome"] == Outcome.JUDGE_ERROR for s in scores)

    return compute


@scorer(metrics=[agreement(), calibration_judge_errors()])
def calibration_scorer(items_dir: str):
    items = load_checked(items_dir, release=False)

    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion
        result, details = await _grade(items[state.metadata["item"]], answer)
        agrees = result.outcome.value == target.text
        return Score(
            value=1 if agrees else 0,
            answer=answer,
            explanation=f"expected {target.text}, judge gave {_explain(result)}",
            metadata=details,
        )

    return score


@task
def judge_calibration(cases_file: str = str(CASES_FILE), items_dir: str = str(ITEMS_DIR)) -> Task:
    rows = yaml.safe_load(Path(cases_file).read_text(encoding="utf-8")) or []
    cases = [CalibrationCase.model_validate(row) for row in rows]
    known = load_checked(items_dir, release=False)
    unknown = sorted({c.item for c in cases} - known.keys())
    if unknown:
        raise ValueError(f"calibration cases name unknown items: {unknown}")
    return Task(
        dataset=MemoryDataset(
            [
                Sample(
                    id=c.id,
                    input=c.answer,
                    target=c.expected,
                    metadata={"item": c.item, "answer": c.answer, "why": c.why, "dataset_size": len(cases)},
                )
                for c in cases
            ]
        ),
        solver=fixed_answer(),
        scorer=calibration_scorer(items_dir),
        epochs=1,
    )
