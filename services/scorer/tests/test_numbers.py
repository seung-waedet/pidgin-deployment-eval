from decimal import Decimal

import pytest

from pidgin_scorer.numbers import contains_number, extract_numbers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Total na 20,076.88", [Decimal("20076.88")]),
        ("₦20,076.88 comot", [Decimal("20076.88")]),
        ("NGN20,076.88", [Decimal("20076.88")]),
        ("N37,700.", [Decimal("37700")]),
        ("#37,700", [Decimal("37700")]),
        ("25k", [Decimal("25000")]),
        ("25 K", [Decimal("25000")]),
        ("25 thousand naira", [Decimal("25000")]),
        ("1.5 million", [Decimal("1500000")]),
        ("1.5M", [Decimal("1500000")]),
        ("5m naira", [Decimal("5000000")]),
        ("2 mil", [Decimal("2000000")]),
        ("5 thousands", [Decimal("5000")]),
        ("wait 5 minutes", [Decimal("5")]),
        ("1,000,000", [Decimal("1000000")]),
        ("88 kobo", [Decimal("88")]),
    ],
)
def test_extract_numbers(text, expected):
    assert extract_numbers(text) == expected


def test_a_number_inside_a_larger_one_does_not_count():
    assert extract_numbers("125,000") == [Decimal("125000")]
    assert contains_number("dem take 125,000", Decimal("25000")) is False


def test_k_followed_by_a_word_is_not_thousands():
    assert extract_numbers("25kobo") == [Decimal("25")]


def test_contains_number_compares_values_not_text():
    assert contains_number("Balance na NGN37,700.00", Decimal("37700")) is True
    assert contains_number("Balance na 41,200", Decimal("37700")) is False


def test_spelled_out_numbers_are_not_read():
    assert extract_numbers("twenty thousand") == []


def test_malformed_grouping_is_not_read_as_a_valid_number():
    assert Decimal("25000") not in extract_numbers("25,0000")


def test_no_number_is_read_from_partway_through_another():
    assert Decimal("25000") not in extract_numbers("1,25,000")


def test_a_spaced_m_is_not_millions():
    assert extract_numbers("wait 5 m") == [Decimal("5")]
    assert extract_numbers("5m naira") == [Decimal("5000000")]


def test_a_number_glued_to_a_word_by_a_dot_is_not_read():
    assert extract_numbers("Ref.37700") == []
