import json
from pathlib import Path

import pytest
import yaml
from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from pidgin_scorer.report import main as report_main
from pidgin_scorer.report import is_result, problems, render
from pidgin_scorer.task import ReleaseError, judge_calibration, pidgin_eval


def item(item_id, expected="answer", pair=None, review="approved", auto_fail=()):
    return {
        "id": item_id,
        "family": "bank_alert",
        "expected": expected,
        "review": review,
        "source": "Amt: NGN100.00 CR",
        "question": "Wetin happen?",
        "reference": "NGN100 entered.",
        "criteria": [{"id": "main", "text": "Says NGN100 entered", "critical": True}],
        "auto_fail": list(auto_fail),
        **({"pair": pair} if pair else {}),
    }


def verdict(met: bool, triggered: bool = False, auto_fail: int = 0) -> str:
    return json.dumps(
        {
            "understood_reason": "r",
            "understood": True,
            "criteria": [{"id": "main", "reason": "r", "met": met}],
            "auto_fail": [{"index": i, "reason": "r", "triggered": triggered} for i in range(auto_fail)],
        }
    )


def mock(*replies: str):
    return get_model(
        "mockllm/model",
        memoize=False,
        custom_outputs=[ModelOutput.from_content(model="mockllm/model", content=r) for r in replies],
    )


def write_items(directory: Path, rows: list[dict]) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "bank_alert.yaml").write_text(yaml.safe_dump(rows), encoding="utf-8")
    return str(directory)


def run(task, answers: list[str], verdicts: list[str], tmp_path: Path):
    [log] = eval(
        task,
        model=mock(*answers),
        model_roles={"grader": mock(*verdicts)},
        max_samples=1,
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert log.status == "success", log.error
    return {name: m.value for name, m in log.results.scores[0].metrics.items()}, log


THREE_ITEMS = [
    item("bank-001a", pair="bank-001"),
    item("bank-001b", expected="flag", pair="bank-001"),
    item("bank-002"),
]


def test_pair_and_answer_scores_from_a_full_run(tmp_path):
    items_dir = write_items(tmp_path / "items", THREE_ITEMS)
    metrics, _ = run(
        pidgin_eval(items_dir=items_dir),
        ["NGN100 enter", "I no see am", "no idea"],
        [verdict(True), verdict(True), verdict(False)],
        tmp_path,
    )
    assert metrics == {
        "item_accuracy": pytest.approx(2 / 3),
        "answer_accuracy": 0.5,
        "pair_score": 1.0,
        "judge_errors": 0,
    }


def test_a_model_that_flags_everything_gets_no_pair_credit(tmp_path):
    items_dir = write_items(tmp_path / "items", THREE_ITEMS)
    metrics, _ = run(
        pidgin_eval(items_dir=items_dir),
        ["I no fit tell"] * 3,
        [verdict(False), verdict(True), verdict(False)],
        tmp_path,
    )
    assert (metrics["pair_score"], metrics["answer_accuracy"]) == (0.0, 0.0)


def test_unreadable_judge_output_is_retried_then_counted_as_a_judge_error(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002")])
    metrics, log = run(
        pidgin_eval(items_dir=items_dir),
        ["NGN100 enter"],
        ["not json", "still not json", "nope"],
        tmp_path,
    )
    assert metrics["judge_errors"] == 1
    assert len(log.samples[0].scores["pidgin_scorer"].metadata["judge_raw"]) == 3


def test_judge_retry_recovers_on_a_later_attempt(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002")])
    metrics, _ = run(pidgin_eval(items_dir=items_dir), ["NGN100 enter"], ["oops", verdict(True)], tmp_path)
    assert (metrics["item_accuracy"], metrics["judge_errors"]) == (1.0, 0)


def test_triggered_auto_fail_fails_the_item(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002", auto_fail=["Asks for a PIN."])])
    metrics, _ = run(
        pidgin_eval(items_dir=items_dir),
        ["Send your PIN"],
        [verdict(True, triggered=True, auto_fail=1)],
        tmp_path,
    )
    assert metrics["item_accuracy"] == 0.0


def test_real_runs_refuse_draft_items(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002", review="draft")])
    with pytest.raises(ReleaseError, match="bank-002"):
        pidgin_eval(items_dir=items_dir)
    pidgin_eval(items_dir=items_dir, release=False)


def test_the_model_sees_only_the_source_and_question(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002")])
    _, log = run(pidgin_eval(items_dir=items_dir), ["x"], [verdict(True)], tmp_path)
    [message] = log.samples[0].messages[:1]
    assert message.role == "user"
    assert message.text == "Amt: NGN100.00 CR\n\nWetin happen?"


def test_calibration_reports_agreement_with_the_expected_verdicts(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002")])
    cases = tmp_path / "cases.yaml"
    cases.write_text(
        yaml.safe_dump(
            [
                {"id": "c1", "item": "bank-002", "answer": "NGN100 enter", "expected": "pass", "why": "w", "review": "draft"},
                {"id": "c2", "item": "bank-002", "answer": "I no know", "expected": "fail", "why": "w", "review": "draft"},
            ]
        ),
        encoding="utf-8",
    )
    metrics, _ = run(
        judge_calibration(cases_file=str(cases), items_dir=items_dir),
        [],
        [verdict(True), verdict(True)],
        tmp_path,
    )
    assert metrics == {"agreement": 0.5, "calibration_judge_errors": 0}


def test_calibration_rejects_cases_for_unknown_items(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002")])
    cases = tmp_path / "cases.yaml"
    cases.write_text(
        yaml.safe_dump([{"id": "c1", "item": "bank-999", "answer": "a", "expected": "pass", "why": "w", "review": "draft"}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bank-999"):
        judge_calibration(cases_file=str(cases), items_dir=items_dir)


def test_report_marks_draft_runs_and_shows_reasons(tmp_path):
    items_dir = write_items(
        tmp_path / "items",
        [item("bank-001a", pair="bank-001", review="draft"), item("bank-001b", expected="flag", pair="bank-001", review="draft")],
    )
    _, log = run(
        pidgin_eval(items_dir=items_dir, release=False),
        ["NGN100 enter", "NGN100 enter"],
        [verdict(True), verdict(False)],
        tmp_path,
    )
    table = render([log])
    assert table.startswith("**Development run on draft items. These are not results.**")
    assert "| mockllm/model | 1/2 | 1/1 | 0/1 | 0 |" in table
    assert "| bank-001b | flag | fail (main) |" in table


def failing_on(bad_input: str, reply: str):
    def respond(input, tools, tool_choice, config):
        if bad_input in input[-1].text:
            raise RuntimeError("upstream 429")
        return ModelOutput.from_content(model="mockllm/model", content=reply)

    return get_model("mockllm/model", memoize=False, custom_outputs=respond)


def run_partial(task, model, verdicts, tmp_path, **kwargs):
    [log] = eval(
        task,
        model=model,
        model_roles={"grader": mock(*verdicts)},
        max_samples=1,
        log_dir=str(tmp_path / "logs"),
        display="none",
        fail_on_error=False,
        **kwargs,
    )
    if log.results is None:
        return {}, log
    return {name: m.value for name, m in log.results.scores[0].metrics.items()}, log


def test_errored_samples_make_metrics_and_report_incomplete(tmp_path):
    rows = [item("bank-001a", pair="bank-001"), item("bank-001b", expected="flag", pair="bank-001")]
    rows[1]["question"] = "BAD QUESTION"
    items_dir = write_items(tmp_path / "items", rows)
    metrics, log = run_partial(pidgin_eval(items_dir=items_dir), failing_on("BAD QUESTION", "ok"), [verdict(True)], tmp_path)
    assert metrics["item_accuracy"] != metrics["item_accuracy"]
    table = render([log])
    assert "incomplete: 1 samples errored (bank-001b)" in table


def test_subset_runs_are_incomplete(tmp_path):
    items_dir = write_items(tmp_path / "items", THREE_ITEMS)
    metrics, log = run_partial(pidgin_eval(items_dir=items_dir), mock("I no fit tell"), [verdict(True)], tmp_path, sample_id="bank-001b")
    assert metrics["pair_score"] != metrics["pair_score"]
    assert "incomplete: 2 items not run (bank-001a, bank-002)" in render([log])


def test_multi_epoch_runs_are_refused(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002")])
    _, log = run_partial(pidgin_eval(items_dir=items_dir), mock("a", "b"), [verdict(True)] * 2, tmp_path, epochs=2)
    assert "incomplete: epochs is 2, pair scoring needs 1" in render([log])


def test_report_does_not_crash_on_a_calibration_log(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002")])
    cases = tmp_path / "cases.yaml"
    cases.write_text(
        yaml.safe_dump([{"id": "c1", "item": "bank-002", "answer": "a", "expected": "pass", "why": "w", "review": "draft"}]),
        encoding="utf-8",
    )
    _, log = run(judge_calibration(cases_file=str(cases), items_dir=items_dir), [], [verdict(True)], tmp_path)
    assert "incomplete: not a pidgin_eval log" in render([log])


def test_judge_errors_make_the_run_not_a_result(tmp_path):
    items_dir = write_items(tmp_path / "items", THREE_ITEMS)
    metrics, log = run(
        pidgin_eval(items_dir=items_dir),
        ["I no fit tell"] * 3,
        [verdict(False), verdict(True), "junk", "junk", "junk"],
        tmp_path,
    )
    assert metrics["item_accuracy"] != metrics["item_accuracy"]
    assert metrics["judge_errors"] == 1
    assert "incomplete: 1 judge errors (bank-002), rescore with inspect score" in render([log])
    assert report_main([log.location]) == 1


def test_report_shows_flags_whose_twin_failed(tmp_path):
    items_dir = write_items(tmp_path / "items", THREE_ITEMS)
    _, log = run(
        pidgin_eval(items_dir=items_dir),
        ["I no fit tell"] * 3,
        [verdict(False), verdict(True), verdict(False)],
        tmp_path,
    )
    assert "| mockllm/model | 1/3 | 0/2 | 0/1 | 1 |" in render([log])


FACT_ITEM = item("bank-002") | {"facts": [{"name": "amount", "value": "100"}]}


def test_the_models_answer_is_graded_not_the_answer_key(tmp_path):
    items_dir = write_items(tmp_path / "items", [FACT_ITEM])
    metrics, log = run(pidgin_eval(items_dir=items_dir), ["I no know"], [verdict(True)], tmp_path)
    assert metrics["item_accuracy"] == 0.0
    assert log.samples[0].scores["pidgin_scorer"].metadata["grade"]["missing_facts"] == ["amount"]


def test_the_judge_model_is_recorded_with_every_grade(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002")])
    _, log = run(pidgin_eval(items_dir=items_dir), ["x"], [verdict(True)], tmp_path)
    assert log.samples[0].scores["pidgin_scorer"].metadata["judge_model"] == "mockllm/model"


def test_later_epochs_are_refused_by_the_scorer_itself(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002")])
    _, log = run_partial(pidgin_eval(items_dir=items_dir), mock("a", "b"), [verdict(True)] * 2, tmp_path, epochs=2)
    errors = [s.error.message for s in log.samples if s.error]
    assert errors and all("--epochs 1" in e for e in errors)


def test_samples_that_hit_a_run_limit_make_the_report_incomplete(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002")])
    _, log = run_partial(pidgin_eval(items_dir=items_dir), mock("x"), [verdict(True)], tmp_path, token_limit=1)
    assert "1 samples hit a run limit (bank-002)" in render([log])


def test_report_refuses_a_failed_run_and_a_log_without_an_item_list(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002")])
    _, log = run(pidgin_eval(items_dir=items_dir), ["x"], [verdict(True)], tmp_path)
    failed = log.model_copy(update={"status": "error"})
    assert "run status is error" in problems(failed)
    no_list = log.model_copy(update={"eval": log.eval.model_copy(update={"metadata": None})})
    assert "log has no item list" in problems(no_list)


def test_report_refuses_logs_with_different_item_sets(tmp_path):
    a = write_items(tmp_path / "a", [item("bank-002")])
    b = write_items(tmp_path / "b", [item("bank-003")])
    _, log_a = run(pidgin_eval(items_dir=a), ["x"], [verdict(True)], tmp_path / "ra")
    _, log_b = run(pidgin_eval(items_dir=b), ["x"], [verdict(True)], tmp_path / "rb")
    assert "logs cover different item sets" in render([log_a, log_b])


def test_report_exit_code(tmp_path, capsys):
    items_dir = write_items(tmp_path / "items", [item("bank-002")])
    _, complete = run(pidgin_eval(items_dir=items_dir), ["x"], [verdict(True)], tmp_path / "ok")
    assert report_main([complete.location]) == 0
    _, partial = run_partial(pidgin_eval(items_dir=items_dir), mock("x"), [verdict(True)], tmp_path / "bad", token_limit=1)
    assert report_main([partial.location]) == 1


def test_report_exits_nonzero_on_draft_runs(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002", review="draft")])
    _, log = run(pidgin_eval(items_dir=items_dir, release=False), ["x"], [verdict(True)], tmp_path)
    assert report_main([log.location]) == 1


def test_report_exits_nonzero_on_logs_with_different_item_sets(tmp_path):
    a = write_items(tmp_path / "a", [item("bank-002")])
    b = write_items(tmp_path / "b", [item("bank-003")])
    _, log_a = run(pidgin_eval(items_dir=a), ["x"], [verdict(True)], tmp_path / "ra")
    _, log_b = run(pidgin_eval(items_dir=b), ["x"], [verdict(True)], tmp_path / "rb")
    assert report_main([log_a.location, log_b.location]) == 1


def test_report_refuses_a_task_that_only_ends_with_the_name(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002")])
    _, log = run(pidgin_eval(items_dir=items_dir), ["x"], [verdict(True)], tmp_path)
    renamed = log.model_copy(update={"eval": log.eval.model_copy(update={"task": "my_pidgin_eval"})})
    assert problems(renamed) == ["not a pidgin_eval log (my_pidgin_eval)"]


def test_calibration_agreement_is_nan_on_a_partial_run(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002")])
    cases = tmp_path / "cases.yaml"
    rows = [{"id": f"c{i}", "item": "bank-002", "answer": "a", "expected": "pass", "why": "w", "review": "draft"} for i in (1, 2)]
    cases.write_text(yaml.safe_dump(rows), encoding="utf-8")
    metrics, _ = run_partial(judge_calibration(cases_file=str(cases), items_dir=items_dir), mock(), [verdict(True)], tmp_path, limit=1)
    assert metrics["agreement"] != metrics["agreement"]


def test_calibration_agreement_leaves_out_judge_errors(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002")])
    cases = tmp_path / "cases.yaml"
    rows = [{"id": f"c{i}", "item": "bank-002", "answer": "a", "expected": "pass", "why": "w", "review": "draft"} for i in (1, 2)]
    cases.write_text(yaml.safe_dump(rows), encoding="utf-8")
    metrics, _ = run(judge_calibration(cases_file=str(cases), items_dir=items_dir), [], [verdict(True), "x", "x", "x"], tmp_path)
    assert metrics == {"agreement": 1.0, "calibration_judge_errors": 1}


def tampered(log, sample_id, **score_metadata):
    copy = log.model_copy(deep=True)
    for sample in copy.samples:
        if sample.id == sample_id:
            sample.scores["pidgin_scorer"].metadata.update(score_metadata)
    return copy


def test_report_refuses_grades_made_against_edited_items(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002")])
    _, log = run(pidgin_eval(items_dir=items_dir), ["x"], [verdict(True)], tmp_path)
    assert problems(log) == []
    assert problems(tampered(log, "bank-002", item_hash="0" * 16)) == [
        "1 items graded against content that differs from the run (bank-002)"
    ]


def test_a_rescore_against_draft_items_is_not_a_result(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002")])
    _, log = run(pidgin_eval(items_dir=items_dir), ["x"], [verdict(True)], tmp_path)
    assert is_result([log])
    assert not is_result([tampered(log, "bank-002", review="draft")])


def test_logs_with_the_same_ids_but_different_content_are_not_comparable(tmp_path):
    a = write_items(tmp_path / "a", [item("bank-002")])
    b = write_items(tmp_path / "b", [item("bank-002") | {"question": "Something else?"}])
    _, log_a = run(pidgin_eval(items_dir=a), ["x"], [verdict(True)], tmp_path / "ra")
    _, log_b = run(pidgin_eval(items_dir=b), ["x"], [verdict(True)], tmp_path / "rb")
    assert "logs cover different item sets" in render([log_a, log_b])
    assert not is_result([log_a, log_b])


def test_logs_marked_by_different_judges_are_not_comparable(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002")])
    _, log_a = run(pidgin_eval(items_dir=items_dir), ["x"], [verdict(True)], tmp_path / "ra")
    _, log_b = run(pidgin_eval(items_dir=items_dir), ["x"], [verdict(True)], tmp_path / "rb")
    other_judge = tampered(log_b, "bank-002", judge_model="openrouter/some/other")
    assert "logs were marked by different judges" in render([log_a, other_judge])
    assert not is_result([log_a, other_judge])


def test_an_invalidated_log_is_not_a_result(tmp_path):
    items_dir = write_items(tmp_path / "items", [item("bank-002")])
    _, log = run(pidgin_eval(items_dir=items_dir), ["x"], [verdict(True)], tmp_path)
    assert "log is marked invalidated" in problems(log.model_copy(update={"invalidated": True}))


@pytest.mark.parametrize(
    "output",
    [
        ModelOutput.from_content(model="mockllm/model", content=""),
        ModelOutput.from_content(model="mockllm/model", content="Your balance na", stop_reason="max_tokens"),
    ],
)
def test_empty_or_truncated_answers_error_instead_of_failing(tmp_path, output):
    items_dir = write_items(tmp_path / "items", [item("bank-002")])
    model = get_model("mockllm/model", memoize=False, custom_outputs=[output])
    _, log = run_partial(pidgin_eval(items_dir=items_dir), model, [verdict(False)], tmp_path)
    assert log.samples[0].error is not None
    assert report_main([log.location]) == 1
