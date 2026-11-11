# dataset

Schema and checks for test items. Items are YAML files, one per task family, so a
reviewer can correct them in a text editor.

Fields that need explaining:

- `expected` is `answer` when the source settles the question, and `flag` for a
  trap where it cannot, so the right response is to say so.
- `pair` links a trap to its answerable twin. A pair has one of each.
- `facts` are numbers the question asks for. List one only when an answer without
  it would be wrong. The planned scorer will compare numbers, not substrings, so
  `25,000` will not match inside `125,000`.
- `criteria` will be marked by an LLM judge (not built yet). `critical: true`
  means failing it fails the item.
- `auto_fail` lists mistakes that fail the item whatever else is right.
- `review` starts as `draft`. Only the native-speaker reviewer sets `approved`.

`uv run pidgin-dataset check <dir> --release` fails on any draft.
