# llm

An Inspect AI model provider that runs prompts through the local Claude Code CLI,
so any Inspect task can use Claude as `claude-code/<model>` with no API key. The
model name is whatever `claude --model` accepts, for example `claude-code/fable`
or `claude-code/opus`.

This project uses it for the LLM judge. Calls always use the Claude Code sign-in
and count against that subscription. `ANTHROPIC_API_KEY` is deliberately not passed
to the CLI, so a key set in the shell can never turn judge calls into an API bill.
One gap remains: an API key saved inside Claude Code's own config (`claude` login
with a key, stored under the user's home) would still be used. Check with
`claude /status` if that matters.

## Isolation

Each call runs `claude --print` in an empty temporary directory with:

- `--setting-sources ""`, so the user's CLAUDE.md, memory, hooks and settings are
  not loaded into the judge's context
- `--system-prompt`, replacing Claude Code's own agent prompt
- `--tools ""` and `--disable-slash-commands`, so the call is a plain completion
  with no agent loop
- an allowlisted environment, so settings of a Claude Code session the eval was
  launched from (effort, base URL and others) do not reach the judge

A live test measures this: the judge's context is about 955 tokens isolated, against
10,246 with the user's settings loaded.

A reply that is really an error raises instead of being returned as an answer.
The checks are `is_error`, a message type other than `result`, a non-success
`subtype`, an empty result, a known in-band error message such as a usage limit,
and, as the catch-all for wordings never seen before, a reply that reports zero
generated tokens.

## Limits

- One user message plus optional system messages. Multi-turn input and tool calls
  raise an error.
- No temperature control. The CLI has no flag for it, so outputs can vary between
  runs.
- Inspect's adaptive concurrency decides how many calls run at once. Pass
  `--max-connections N` to cap it.

## Tests

```bash
uv run pytest            # gate tests, use a fake claude binary
uv run pytest -m live    # one real call, checks no user context leaks in
```

`CLAUDE_CODE_BIN` points the provider at a different binary. The tests use it.
