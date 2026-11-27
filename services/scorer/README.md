# scorer

The Inspect AI task that runs models over the items, the LLM judge, and pair
scoring.

## How an answer is graded

An answer passes only if all three hold:

1. Every `fact` number appears in it. Code checks this, comparing numbers as values
   (`25k`, `NGN25,000.00` and `25 thousand` are all 25000, `1.5m` is 1500000, and
   `125,000` is not 25000). A bare `m` counts as million only straight after the
   number, so "wait 5 m" stays 5. Not read: spelled-out numbers ("twenty thousand"),
   space or dot grouping (`52 850`, `52.850`), and numbers glued to a word by a
   dot (`Ref.52850`). A fact written those ways shows as a missing fact.
2. Every `critical` criterion is met, as judged by the LLM judge.
3. No `auto_fail` is triggered, as judged by the LLM judge.

The judge also labels whether the answer understood the question at all,
separately from whether it was right.

If the judge's reply cannot be read after 3 attempts, the item is a `judge_error`,
unless a fact is missing, in which case code has already failed it. A run with any
judge error is not a result: dropping those items would shrink the denominator, and
the judge may fail more often on exactly the long, messy answers that are wrong.
Rescore the log with `inspect score --action overwrite <log>`. The default
`append` action adds a second scorer column that the report does not read.

An empty answer, or one cut off by `max_tokens`, errors the sample instead of
failing it. On free endpoints those are more often upstream failures than the
model's real reply, so the sample should be rerun.

## Pair scoring

A trap item and its answerable twin form a pair. A pair passes only if both pass.
Most traps' critical criteria require the answer to say what the text does and
does not show, so a bare refusal or a generic "contact your bank" should fail them
too. Where a trap's correct answer is itself cautious, a generic hedge can pass the
trap. The twin is there for the model that hedges where the text does support an
answer: it passes the trap and fails the twin, so it earns no pair credit. How well
each twin does this depends on its rubric, and the calibration set does not yet
test refusals or hedges on traps. The report's "flagged,
twin failed" column counts those pairs.

## Metrics and the report

The in-run metrics are `item_accuracy`, `answer_accuracy` and `pair_score`, plus a
`judge_errors` count. The three scores are NaN unless every item in the dataset was
graded with no judge error, because Inspect silently drops errored samples before
metrics and accepts `--limit` and `--sample-id` subsets. Calibration's `agreement`
is NaN on a partial run or with any judge error in the same way.

`pidgin-report` is the result of record. It refuses to summarise a log that is not
a complete `pidgin_eval` run: wrong task, failed status, errored samples, items not
run, samples that hit a run limit, any judge error, more than one epoch, an
invalidated log, or grades made against item content that differs from the run.
Across logs it also refuses different item sets (compared by content fingerprint,
not just id) and different judges. It prints why instead, and exits with status 1.
It also exits 1 on draft items, which are never a result. For a complete run it
shows, per model:

- items passed
- answerable items passed
- pairs passed
- traps passed whose twin failed, a sign the model hedged where the text supported
  an answer

Pair scoring needs one grade per item, so the scorer raises on any epoch after the
first. With `--fail-on-error` off, a multi-epoch run's in-run metrics still show
the first epoch's numbers. The report refuses such a run.

Every grade records which judge model produced it (`judge_model` in the score
metadata), because Inspect leaves the default grader out of the log. For the
default this is the alias `claude-code/fable`, so a run before and after Claude
Code moves `fable` to a newer model would look like the same judge.

Every sample records a fingerprint of its item at run time, and every grade
records the fingerprint of the item it was graded against. They must match, so
editing a rubric and rescoring cannot produce a result nobody ran.

The judge's reply must be a single JSON object, optionally in a code fence. A
verdict pulled out of surrounding prose could be one the judge was only quoting.

## Running

The judge defaults to `claude-code/fable` through the local Claude Code CLI
(see `services/llm`). Override it with `--model-role grader=<model>`.

```bash
# development run on draft items
uv run inspect eval pidgin_scorer/task.py@pidgin_eval -T release=false \
  --model openrouter/google/gemma-4-31b-it:free

# real run: refuses while any item is still draft
uv run inspect eval pidgin_scorer/task.py@pidgin_eval --model <model>

# how often the judge agrees with the labelled calibration cases
uv run inspect eval pidgin_scorer/task.py@judge_calibration --model none

# markdown table across models
uv run pidgin-report logs/*.eval
```

The model under test gets the source text and the question as one user message,
with no system prompt, as a real person would paste it. The one exception is a
`claude-code/*` model under test, which gets "You are a helpful assistant." because
Claude Code always replaces its own agent prompt with something.

## Tests and evals

- `uv run pytest` runs the gate tests offline, with Inspect's mock model as both
  the model under test and the judge.
- `judge_calibration` is the periodic eval of the judge itself. It costs judge
  calls and is not part of the gate. Its cases in `items/judge/cases.yaml` include
  answers that try to inject instructions or a fake verdict into the judge, on items
  with no number checks, so only the judge can catch them.
