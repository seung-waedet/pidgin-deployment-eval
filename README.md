# Pidgin deployment-readiness eval

If you put an LLM in front of people in Nigeria who read and write Pidgin, does it
give correct, safe, usable answers on everyday tasks?

This project tests that with hand-written items: reading a medication label,
explaining a bank alert or POS reversal, and summarising a public notice.

**Status: test items are being written. No models have been run and there are no
results yet.**

## What is measured

A model sees a real-world style text and a question in Pidgin. A native speaker
wrote the answer key and the marking rubric for each item.

Trap items are mixed into every task family. In a trap, the text does not contain
the answer, and the right response is to say so. Every trap has an answerable twin
built from the same scenario. A model only earns trap credit if it also answers the
twin, so a model that refuses everything scores nothing.

## Not medical or financial advice

Medication items test whether a model conveys what a label says. They do not test
medical judgment, and nothing here is medical advice. Bank items test whether a
model reads an alert correctly. They are not financial advice. A high score does
not mean a model is safe to use for health or money decisions.

## Layout

- `services/dataset` holds the item schema and the checks every item must pass.
- `docs/related-work.md` covers what existing work measures and how this differs.

Test items live in a separate private repo until release. Published items go to
Hugging Face. A held-out share is never published, so the leaderboard can be
rechecked on items no model could have trained on.

## Run the checks

```bash
cd services/dataset
uv run pytest
uv run pidgin-dataset check <items-directory>
```

Add `--release` to fail while any item is still a draft. A results run must pass
this check, so no unreviewed item can reach the leaderboard.
