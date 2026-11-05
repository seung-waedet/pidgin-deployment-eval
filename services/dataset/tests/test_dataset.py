from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pidgin_dataset.cli import main
from pidgin_dataset.schema import DatasetError, Item, check_dataset, load_items


def item(**overrides) -> dict:
    return {
        "id": "bank-001",
        "family": "bank_alert",
        "expected": "answer",
        "review": "approved",
        "source": "CR: NGN100.00",
        "question": "Wetin be this alert?",
        "reference": "NGN100 entered the account.",
        "criteria": [{"id": "amount", "text": "States NGN100", "critical": True}],
    } | overrides


def trap_pair() -> list[dict]:
    return [
        item(id="bank-001a", pair="bank-001"),
        item(id="bank-001b", pair="bank-001", expected="flag"),
    ]


def write_items(directory: Path, rows: list[dict]) -> None:
    (directory / "bank_alert.yaml").write_text(yaml.safe_dump(rows), encoding="utf-8")


def test_item_parses_a_valid_row():
    parsed = Item.model_validate(item())
    assert parsed.expected == "answer"
    assert parsed.criteria[0].critical is True


def test_item_rejects_an_unknown_field():
    with pytest.raises(ValidationError):
        Item.model_validate(item(typo_field="x"))


def test_item_needs_at_least_one_criterion():
    with pytest.raises(ValidationError):
        Item.model_validate(item(criteria=[]))


def test_trap_without_a_twin_id_is_invalid():
    with pytest.raises(ValidationError, match="must name its pair"):
        Item.model_validate(item(expected="flag"))


def test_duplicate_ids_are_rejected():
    items = [Item.model_validate(item()), Item.model_validate(item())]
    with pytest.raises(DatasetError, match="duplicate id: bank-001"):
        check_dataset(items)


def test_pair_needs_one_answer_and_one_trap():
    both_answer = [item(id="bank-001a", pair="bank-001"), item(id="bank-001b", pair="bank-001")]
    items = [Item.model_validate(row) for row in both_answer]
    with pytest.raises(DatasetError, match="pair bank-001 needs one answer and one flag"):
        check_dataset(items)


def test_valid_trap_pair_passes():
    check_dataset([Item.model_validate(row) for row in trap_pair()])


def test_check_passes_on_drafts_by_default(tmp_path, capsys):
    write_items(tmp_path, [item(review="draft")])
    assert main(["check", str(tmp_path)]) == 0
    assert "1 items" in capsys.readouterr().out


def test_release_check_fails_while_any_item_is_a_draft(tmp_path, capsys):
    write_items(tmp_path, [item(review="draft")])
    assert main(["check", str(tmp_path), "--release"]) == 1
    assert "1 items still draft" in capsys.readouterr().err


def test_release_check_passes_when_all_items_are_approved(tmp_path):
    write_items(tmp_path, trap_pair())
    assert main(["check", str(tmp_path), "--release"]) == 0


def test_load_items_reads_every_yaml_file(tmp_path):
    write_items(tmp_path, [item()])
    assert [i.id for i in load_items(tmp_path)] == ["bank-001"]
