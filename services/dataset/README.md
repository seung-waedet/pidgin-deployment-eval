# dataset

Schema and checks for test items. Items are YAML files, one per task family, so a
reviewer can correct them in a text editor.

Fields that need explaining:

- `expected` is `answer` when the source settles the question, and `flag` for a
  trap where it cannot, so the right response is to say so.
- `pair` links a trap to its answerable twin. A pair has one of each.
- `facts` are numbers the question asks for. List one only when an answer without
  it would be wrong. The scorer compares numbers, not substrings, so `25,000` does
  not match inside `125,000`.
- `criteria` are marked by the LLM judge. `critical: true` means failing it fails
  the item.
- `auto_fail` lists mistakes that fail the item whatever else is right.
- `review` starts as `draft`. Only the native-speaker reviewer sets `approved`.

`uv run pidgin-dataset check <dir> --release` fails on any draft.
