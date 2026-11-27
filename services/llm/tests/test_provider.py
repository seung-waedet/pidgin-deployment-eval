import asyncio
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model

from inspect_ai.tool import ToolInfo

from claude_code_llm import provider
from claude_code_llm.provider import (
    DEFAULT_SYSTEM_PROMPT,
    ClaudeCodeError,
    build_command,
    child_environment,
    split_messages,
)

pytestmark = pytest.mark.anyio


def fake_claude(tmp_path: Path, payload: dict | str, exit_code: int = 0, sleep: float = 0) -> Path:
    """A stand-in claude binary that records its call, optionally hangs, then replies."""
    if isinstance(payload, dict) and "usage" not in payload:
        payload = payload | {"usage": {"input_tokens": 10, "output_tokens": 5}}
    reply = payload if isinstance(payload, str) else json.dumps(payload)
    script = tmp_path / "claude"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        f"open({str(tmp_path / 'call.json')!r}, 'w').write(json.dumps("
        "{'argv': sys.argv[1:], 'stdin': sys.stdin.read(), 'cwd': os.getcwd(),"
        " 'env_names': sorted(os.environ), 'pid': os.getpid()}))\n"
        f"time.sleep({sleep})\n"
        f"sys.stdout.write({reply!r})\n"
        f"sys.exit({exit_code})\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def recorded_call(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "call.json").read_text())


def test_command_isolates_the_session_from_user_settings():
    command = build_command("claude", "fable", "Mark this.")
    assert command[command.index("--setting-sources") + 1] == ""
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--system-prompt") + 1] == "Mark this."
    for flag in ["--print", "--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands"]:
        assert flag in command
    assert command[command.index("--output-format") + 1] == "json"


def test_split_messages_joins_system_and_takes_the_user_prompt():
    system, prompt = split_messages(
        [ChatMessageSystem(content="a"), ChatMessageSystem(content="b"), ChatMessageUser(content="q")]
    )
    assert (system, prompt) == ("a\n\nb", "q")


@pytest.mark.parametrize(
    "messages",
    [
        [ChatMessageUser(content="q"), ChatMessageAssistant(content="a")],
        [ChatMessageUser(content="q"), ChatMessageAssistant(content="a"), ChatMessageUser(content="q2")],
        [ChatMessageSystem(content="s")],
    ],
)
def test_split_messages_rejects_anything_but_one_user_message(messages):
    with pytest.raises(ClaudeCodeError, match="one user message"):
        split_messages(messages)


async def test_generate_sends_the_prompt_on_stdin_and_returns_the_result(tmp_path, monkeypatch):
    usage = {"input_tokens": 12, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 40, "output_tokens": 4}
    payload = {"is_error": False, "result": "Na 52,850.", "usage": usage}
    monkeypatch.setenv("CLAUDE_CODE_BIN", str(fake_claude(tmp_path, payload)))

    output = await get_model("claude-code/fable").generate("Which balance?")

    assert output.completion == "Na 52,850."
    assert (output.usage.input_tokens_cache_read, output.usage.total_tokens) == (900, 956)
    call = recorded_call(tmp_path)
    assert call["stdin"] == "Which balance?"
    assert call["argv"][call["argv"].index("--model") + 1] == "fable"
    assert call["cwd"] != str(Path.cwd())


async def test_reported_errors_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_BIN", str(fake_claude(tmp_path, {"is_error": True, "result": "quota"})))
    with pytest.raises(ClaudeCodeError, match="quota"):
        await get_model("claude-code/fable").generate("q")


async def test_a_reply_without_a_result_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_BIN", str(fake_claude(tmp_path, {"is_error": False})))
    with pytest.raises(ClaudeCodeError, match="no result"):
        await get_model("claude-code/fable").generate("q")


@pytest.mark.parametrize(
    "payload",
    [
        "[1, 2]",
        {"is_error": False, "result": "Claude AI usage limit reached|1764200000"},
        {"is_error": False, "result": ""},
        {"is_error": False, "subtype": "error_during_execution", "result": "partial"},
        {"is_error": False, "result": "API Error: 529 overloaded"},
        {"is_error": False, "result": "Invalid API key · Please run /login"},
        {"is_error": False, "result": "You've hit your limit · resets 3pm"},
        {"is_error": False, "result": "\nAPI Error: 529 overloaded"},
        {"type": "system", "is_error": False, "result": "not a result message"},
        {"is_error": False, "result": "   "},
        {"is_error": False, "result": "Not logged in · Please run /login", "usage": {"output_tokens": 0}},
        {"is_error": False, "result": "Some brand new error wording", "usage": {"input_tokens": 0}},
        {"is_error": False, "result": "no usage block at all", "usage": None},
    ],
)
async def test_replies_that_are_not_answers_raise(tmp_path, monkeypatch, payload):
    monkeypatch.setenv("CLAUDE_CODE_BIN", str(fake_claude(tmp_path, payload)))
    with pytest.raises(ClaudeCodeError):
        await get_model("claude-code/fable").generate("q")


async def test_a_truncated_reply_keeps_its_stop_reason(tmp_path, monkeypatch):
    payload = {"is_error": False, "subtype": "success", "result": "half an", "stop_reason": "max_tokens"}
    monkeypatch.setenv("CLAUDE_CODE_BIN", str(fake_claude(tmp_path, payload)))
    output = await get_model("claude-code/fable").generate("q")
    assert output.stop_reason == "max_tokens"


async def test_a_refusal_is_reported_as_content_filtered(tmp_path, monkeypatch):
    payload = {"is_error": False, "result": "I can't help with that.", "stop_reason": "refusal"}
    monkeypatch.setenv("CLAUDE_CODE_BIN", str(fake_claude(tmp_path, payload)))
    output = await get_model("claude-code/fable").generate("q")
    assert output.stop_reason == "content_filter"


async def test_tool_calls_are_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_BIN", str(fake_claude(tmp_path, {"result": "x"})))
    with pytest.raises(ClaudeCodeError, match="tool calls"):
        await get_model("claude-code/fable").generate("q", tools=[ToolInfo(name="t", description="d")])


async def test_a_plain_prompt_gets_the_default_system_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_BIN", str(fake_claude(tmp_path, {"result": "x"})))
    await get_model("claude-code/fable").generate("q")
    argv = recorded_call(tmp_path)["argv"]
    assert argv[argv.index("--system-prompt") + 1] == DEFAULT_SYSTEM_PROMPT


async def test_the_default_timeout_applies_when_none_is_set(tmp_path, monkeypatch):
    monkeypatch.setattr(provider, "DEFAULT_TIMEOUT_SECONDS", 1)
    monkeypatch.setenv("CLAUDE_CODE_BIN", str(fake_claude(tmp_path, {"result": "late"}, sleep=30)))
    with pytest.raises(ClaudeCodeError, match="within 1s"):
        await get_model("claude-code/fable").generate("q")


def test_only_allowlisted_variables_reach_the_child():
    environ = {name: "v" for name in ["PATH", "HOME", "LC_ALL", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY",
                                      "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE", "CLAUDE_EFFORT", "OTHER",
                                      "ANTHROPIC_MODEL", "NODE_OPTIONS"]}
    assert set(child_environment(environ)) == {"PATH", "HOME", "LC_ALL", "CLAUDE_CODE_OAUTH_TOKEN"}


async def gone_within(pid: int, seconds: float) -> bool:
    """True once pid no longer exists. Polls, so a busy machine cannot fail it early."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        await asyncio.sleep(0.05)
    return False


async def test_a_timeout_holds_even_when_a_child_keeps_the_pipes_open(tmp_path, monkeypatch):
    # A shell script, not Python, so start-up is milliseconds. macOS also scans a new
    # executable on its first run (about 1s, more under load), so run it once first:
    # the timed call must not depend on how fast the machine starts a process.
    script = tmp_path / "claude"
    script.write_text(
        "#!/bin/sh\n"
        '[ "$1" = "--warm" ] && exit 0\n'
        "sleep 47 &\n"
        f"echo $! > {str(tmp_path / 'grandchild')!r}\n"
        "sleep 47\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    subprocess.run([str(script), "--warm"], check=True)
    monkeypatch.setenv("CLAUDE_CODE_BIN", str(script))

    started = time.monotonic()
    with pytest.raises(ClaudeCodeError, match="within 2s"):
        await get_model("claude-code/fable").generate("q", config=GenerateConfig(timeout=2))
    assert time.monotonic() - started < 10

    grandchild = int((tmp_path / "grandchild").read_text())
    assert await gone_within(grandchild, 5), "a grandchild holding the pipes outlived the call"


async def test_a_context_overflow_is_reported_as_cut_off(tmp_path, monkeypatch):
    payload = {"is_error": False, "result": "half", "stop_reason": "model_context_window_exceeded"}
    monkeypatch.setenv("CLAUDE_CODE_BIN", str(fake_claude(tmp_path, payload)))
    output = await get_model("claude-code/fable").generate("q")
    assert output.stop_reason == "model_length"


async def test_a_hung_call_times_out(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_BIN", str(fake_claude(tmp_path, {"result": "late"}, sleep=30)))
    with pytest.raises(ClaudeCodeError, match="did not answer within 1s"):
        await get_model("claude-code/fable").generate("q", config=GenerateConfig(timeout=1))


async def test_a_failing_process_raises_with_its_exit_code(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_BIN", str(fake_claude(tmp_path, "", exit_code=3)))
    with pytest.raises(ClaudeCodeError, match="exited 3"):
        await get_model("claude-code/fable").generate("q")


async def test_non_json_output_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_BIN", str(fake_claude(tmp_path, "not json")))
    with pytest.raises(ClaudeCodeError, match="non-JSON"):
        await get_model("claude-code/fable").generate("q")


async def test_the_session_environment_does_not_leak_into_the_judge(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_BIN", str(fake_claude(tmp_path, {"is_error": False, "result": "ok"})))
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_EFFORT", "max")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://elsewhere.example")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-test-value")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-value")
    monkeypatch.setenv("SOME_UNRELATED_SETTING", "x")

    await get_model("claude-code/fable").generate("q")

    names = set(recorded_call(tmp_path)["env_names"])
    leaked = {"CLAUDECODE", "CLAUDE_EFFORT", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_BIN", "SOME_UNRELATED_SETTING"}
    assert leaked.isdisjoint(names)
    assert "ANTHROPIC_API_KEY" not in names, "the judge must use the Claude Code sign-in, never bill an API key"
    assert {"PATH", "HOME", "CLAUDE_CODE_OAUTH_TOKEN"} <= names


async def test_cancelling_a_call_kills_the_claude_process(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_BIN", str(fake_claude(tmp_path, {"result": "late"}, sleep=30)))
    call = asyncio.create_task(get_model("claude-code/fable").generate("q"))
    while not (tmp_path / "call.json").exists():
        await asyncio.sleep(0.05)
    pid = recorded_call(tmp_path)["pid"]

    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    assert await gone_within(pid, 5), "cancelling the call left the claude process running"


@pytest.mark.live
async def test_live_context_holds_no_user_settings():
    # Measured on 2026-11-27: about 955 context tokens isolated, 10,246 with the
    # user's settings loaded. A leaked CLAUDE.md would push this well past 2,000.
    output = await get_model("claude-code/fable").generate("Reply with the single word ok.")
    usage = output.usage
    context = usage.input_tokens + usage.input_tokens_cache_read + usage.input_tokens_cache_write
    assert context < 2_000, f"judge context is {context} tokens; user settings may be leaking in"
