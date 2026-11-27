# Pidgin deployment-readiness eval

If you put an LLM in front of people in Nigeria who read and write Pidgin, does it
give correct, safe, usable answers on everyday tasks?

This project will test that on everyday tasks. The planned task families are
reading a medication label, explaining a bank alert or POS reversal, and
summarising a public notice.

**Status: 10 bank-alert items are drafted and none have been reviewed yet. The
scorer and LLM judge are built and tested. Only development runs on draft items
have happened, so there are no results yet.**

## What is measured

A model sees a real-world style text and a question in Pidgin. A model drafts each
item; a native speaker corrects its answer key and marking rubric and approves it.
Only approved items will be used.

Every task family will include trap items. In a trap, the text does not contain
the answer, and the right response is to say so. Every trap has an answerable twin
built from the same scenario, and the checks enforce this. A pair only passes if
both twins pass, so a model that refuses everything earns no pair credit.

Answers are marked by an LLM judge (Claude, run through the local Claude Code CLI)
against each item's rubric, plus code checks for required numbers. The judge is
itself tested on labelled answers before its marks are trusted. See
`services/scorer/README.md`.

## Not medical or financial advice

Medication items test whether a model conveys what a label says. They do not test
medical judgment, and nothing here is medical advice. Bank items test whether a
model reads an alert correctly. They are not financial advice. A high score does
not mean a model is safe to use for health or money decisions.

## Layout

- `services/dataset` holds the item schema and the checks every item must pass.
- `services/scorer` is the Inspect AI task, the LLM judge, pair scoring and the
  results report.
- `services/llm` lets Inspect use the local Claude Code CLI as a model, which the
  judge runs on.
- `docs/related-work.md` covers what existing work measures and how this differs.
- `items/` holds the test items. It is a separate private repo cloned into this
  folder and ignored by this one, so items are never committed here.

Items stay private until release so models cannot train on them. The plan is to
publish items on Hugging Face and keep a held-out share unpublished, so results
can be rechecked on items no model could have trained on.

Maintainers with access to the private repo clone it into place:

```bash
git clone https://github.com/seung-waedet/pidgin-deployment-eval-items.git items
```

## Run the checks

```bash
cd services/dataset
uv run pytest
uv run pidgin-dataset check ../../items
```

Add `--release` to fail while any item is still a draft. Real eval runs refuse
draft items in the same way, so no unreviewed item reaches the results.

## Run an eval

Models under test run through OpenRouter. Put `OPENROUTER_API_KEY=...` in a `.env`
file at the repo root (it is gitignored). The judge needs the `claude` CLI signed in.

```bash
cd services/scorer
uv run inspect eval pidgin_scorer/task.py@pidgin_eval --model openrouter/<model>
uv run inspect eval pidgin_scorer/task.py@judge_calibration --model none
uv run pidgin-report logs/*.eval
```

Free OpenRouter models may log prompts or train on them. Never send held-out items
to one.
