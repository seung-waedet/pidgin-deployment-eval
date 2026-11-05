import argparse
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from pidgin_dataset.schema import DatasetError, Review, check_dataset, load_items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pidgin-dataset")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="validate a directory of item files")
    check.add_argument("directory", type=Path)
    check.add_argument("--release", action="store_true", help="also fail on any draft item")
    args = parser.parse_args(argv)

    try:
        items = load_items(args.directory)
        check_dataset(items)
    except (ValidationError, DatasetError) as error:
        print(error, file=sys.stderr)
        return 1

    print(f"{len(items)} items")
    for family, count in sorted(Counter(item.family.value for item in items).items()):
        print(f"  {family}: {count}")
    for expected, count in sorted(Counter(item.expected.value for item in items).items()):
        print(f"  {expected}: {count}")

    drafts = sum(item.review is Review.DRAFT for item in items)
    print(f"  draft: {drafts}")
    if args.release and drafts:
        print(f"{drafts} items still draft", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
