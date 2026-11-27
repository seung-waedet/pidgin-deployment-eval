import argparse
import sys

from inspect_ai.log import EvalLog, read_eval_log

from pidgin_scorer.grade import Grade, Outcome, summarize

SCORER = "pidgin_scorer"
TASK = "pidgin_eval"


def _ids(ids: list[str]) -> str:
    return ", ".join(sorted(ids))


def problems(log: EvalLog) -> list[str]:
    """Reasons this log is not a complete result. Empty means it is one."""
    if log.eval.task.split("/")[-1] != TASK:
        return [f"not a {TASK} log ({log.eval.task})"]
    found = ["log is marked invalidated"] if log.invalidated else []
    if log.status != "success":
        found.append(f"run status is {log.status}")
    epochs = log.eval.config.epochs or 1
    if epochs != 1:
        found.append(f"epochs is {epochs}, pair scoring needs 1")

    samples = log.samples or []
    errored = [str(s.id) for s in samples if s.error or SCORER not in (s.scores or {})]
    if errored:
        found.append(f"{len(errored)} samples errored ({_ids(errored)})")
    limited = [str(s.id) for s in samples if s.limit]
    if limited:
        found.append(f"{len(limited)} samples hit a run limit ({_ids(limited)})")
    expected = set(_item_hashes(log))
    if not expected:
        found.append("log has no item list")
    missing = expected - {str(s.id) for s in samples}
    if missing:
        found.append(f"{len(missing)} items not run ({_ids(list(missing))})")
    # eval-retry reuses finished samples from the original run, so after an item edit
    # one log can mix old and new content. Every sample must match the log's list.
    hashes = _item_hashes(log)
    stale = [str(s.id) for s in samples if (s.metadata or {}).get("item_hash") != hashes.get(str(s.id))]
    if stale:
        found.append(f"{len(stale)} samples ran on content that differs from this log's item list ({_ids(stale)})")
    # A rescore after the items were edited would grade against a rubric no model saw.
    changed = [
        str(s.id)
        for s in samples
        if SCORER in (s.scores or {})
        and s.scores[SCORER].metadata.get("item_hash") != (s.metadata or {}).get("item_hash")
    ]
    if changed:
        found.append(f"{len(changed)} items graded against content that differs from the run ({_ids(changed)})")
    judge_errors = [i for i, g in _grades(log).items() if g.outcome is Outcome.JUDGE_ERROR]
    if judge_errors:
        found.append(f"{len(judge_errors)} judge errors ({_ids(judge_errors)}), rescore with inspect score")
    return found


def _grades(log: EvalLog) -> dict[str, Grade]:
    grades = {}
    for s in log.samples or []:
        score = (s.scores or {}).get(SCORER)
        if score and not s.error:
            grades[str(s.id)] = Grade.model_validate(score.metadata["grade"])
    return grades


def _cell(g: Grade | None) -> str:
    if g is None:
        return "not scored"
    if g.outcome is Outcome.PASS:
        return "pass"
    if g.outcome is Outcome.JUDGE_ERROR:
        return "judge error"
    reasons = g.missing_facts + g.failed_critical + [f"auto fail: {a}" for a in g.triggered_auto_fail]
    return "fail (" + "; ".join(reasons) + ")"


def _item_hashes(log: EvalLog) -> dict[str, str]:
    return (log.eval.metadata or {}).get("item_hashes", {})


def cross_problems(logs: list[EvalLog]) -> list[str]:
    """Reasons these logs cannot be compared side by side. Empty means they can."""
    found = []
    if len({frozenset(_item_hashes(log).items()) for log in logs}) > 1:
        found.append("logs cover different item sets")
    judges = {
        s.scores[SCORER].metadata.get("judge_model")
        for log in logs
        for s in log.samples or []
        if SCORER in (s.scores or {})
    }
    if len(judges) > 1:
        found.append("logs were marked by different judges")
    return found


def _summary_row(model: str, log: EvalLog, grades: dict[str, Grade], shared: list[str]) -> str:
    found = problems(log) + shared
    if found:
        return f"| {model} | incomplete: {'; '.join(found)} |"
    s = summarize(list(grades.values()))
    return (
        f"| {model} | {s.passed}/{s.items} | {s.answer_passed}/{s.answer_items} "
        f"| {s.pairs_passed}/{s.pairs} | {s.flagged_without_credit} |"
    )


def _has_drafts(log: EvalLog) -> bool:
    def draft(sample) -> bool:
        score = (sample.scores or {}).get(SCORER)
        graded_as_draft = score is not None and score.metadata.get("review") == "draft"
        return (sample.metadata or {}).get("review") == "draft" or graded_as_draft

    return any(draft(s) for s in log.samples or [])


def is_result(logs: list[EvalLog]) -> bool:
    """True only for complete, comparable runs graded on approved items."""
    return not cross_problems(logs) and not any(problems(log) or _has_drafts(log) for log in logs)


def render(logs: list[EvalLog]) -> str:
    models = [log.eval.model for log in logs]
    graded = [_grades(log) for log in logs]
    item_ids = sorted({item_id for grades in graded for item_id in grades})
    drafts = any(_has_drafts(log) for log in logs)

    lines = []
    if drafts:
        lines += ["**Development run on draft items. These are not results.**", ""]
    lines += [
        "| model | items passed | answerable passed | pairs passed | flagged, twin failed |",
        "|---|---|---|---|---|",
    ]
    shared = cross_problems(logs)
    lines += [_summary_row(m, log, g, shared) for m, log, g in zip(models, logs, graded)]
    lines += [""]

    lines += ["| item | expected | " + " | ".join(models) + " |"]
    lines += ["|---|---|" + "---|" * len(models)]
    for item_id in item_ids:
        expected = next(g[item_id].expected for g in graded if item_id in g)
        cells = [_cell(g.get(item_id)) for g in graded]
        lines.append(f"| {item_id} | {expected} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pidgin-report", description="Markdown table from eval logs")
    parser.add_argument("logs", nargs="+", help="Inspect log files, one per model")
    args = parser.parse_args(argv)
    logs = [read_eval_log(path) for path in args.logs]
    print(render(logs))
    return 0 if is_result(logs) else 1


if __name__ == "__main__":
    sys.exit(main())
