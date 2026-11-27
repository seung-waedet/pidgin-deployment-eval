import re
from decimal import Decimal

# Spelled-out numbers ("twenty thousand") are not read. A fact written that way
# shows up as a missing fact in the grade, where a human reviewer can see it.
_NUMBER = re.compile(
    r"(?<![\d.,])"
    r"(?P<whole>\d{1,3}(?:,\d{3})+(?!\d)|\d+)"
    r"(?P<fraction>\.\d+)?"
    # A bare m means million only when written straight after the number ("5m"),
    # so "wait 5 m" stays 5.
    r"(?:(?P<adjacent>[km])\b|\s*(?P<scale>k|thousands?|millions?|mil)\b)?",
    re.IGNORECASE,
)

_THOUSAND, _MILLION = Decimal(1_000), Decimal(1_000_000)
_SCALE = {"k": _THOUSAND, "thousand": _THOUSAND, "thousands": _THOUSAND,
          "m": _MILLION, "mil": _MILLION, "million": _MILLION, "millions": _MILLION}


def extract_numbers(text: str) -> list[Decimal]:
    numbers = []
    for match in _NUMBER.finditer(text):
        value = Decimal(match["whole"].replace(",", "") + (match["fraction"] or ""))
        scale = match["adjacent"] or match["scale"]
        if scale:
            value *= _SCALE[scale.lower()]
        numbers.append(value)
    return numbers


def contains_number(text: str, value: Decimal) -> bool:
    return value in extract_numbers(text)
