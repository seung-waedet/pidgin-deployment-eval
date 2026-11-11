# Pidgin deployment-readiness eval

If you put an LLM in front of people in Nigeria who read and write Pidgin, does it
give correct, safe, usable answers on everyday tasks?

This project will test that on everyday tasks. The planned task families are
reading a medication label, explaining a bank alert or POS reversal, and
summarising a public notice.

**Status: the first test items (bank alerts) are drafted and none have been
reviewed yet. No models have been run and there are no results yet.**

## What is measured

A model sees a real-world style text and a question in Pidgin. A model drafts each
item; a native speaker corrects its answer key and marking rubric and approves it.
Only approved items will be used.

Every task family will include trap items. In a trap, the text does not contain
the answer, and the right response is to say so. Every trap has an answerable twin
built from the same scenario, and the checks enforce this. Once scoring is built, a
model will only earn trap credit if it also answers the twin, so a model that
refuses everything will score nothing.

## Not medical or financial advice

Medication items test whether a model conveys what a label says. They do not test
medical judgment, and nothing here is medical advice. Bank items test whether a
model reads an alert correctly. They are not financial advice. A high score does
not mean a model is safe to use for health or money decisions.

## Layout

- `services/dataset` holds the item schema and the checks every item must pass.
- `docs/related-work.md` covers what existing work measures and how this differs.

Test items live in a separate private repo until release. The plan is to publish
items on Hugging Face and keep a held-out share unpublished, so results can be
rechecked on items no model could have trained on.

## Run the checks

```bash
cd services/dataset
uv run pytest
uv run pidgin-dataset check <items-directory>
```

Add `--release` to fail while any item is still a draft. Results runs will have to
pass this check, so no unreviewed item reaches the results.

Not built yet: the scorer, the LLM judge for open answers, and results runs.
