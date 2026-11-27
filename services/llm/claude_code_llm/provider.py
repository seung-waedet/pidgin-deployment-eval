import asyncio
import json
import os
import tempfile
from collections.abc import Mapping

from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.tool import ToolChoice, ToolInfo

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_TIMEOUT_SECONDS = 600

# The CLI can report a failure as an ordinary result with is_error false. These are
# the forms seen or documented; any of them is an outage, never an answer.
_IN_BAND_ERRORS = (
    "Claude AI usage limit reached",
    "API Error:",
    "Invalid API key",
    "You've hit your limit",
)
_STOP_REASONS = {"max_tokens": "max_tokens", "refusal": "content_filter"}


class ClaudeCodeError(RuntimeError):
    pass


def build_command(binary: str, model: str, system_prompt: str) -> list[str]:
    # --setting-sources "" keeps the user's CLAUDE.md, memory and hooks out of the
    # context, and --tools "" makes this a plain text completion with no agent loop.
    return [
        binary,
        "--print",
        "--model", model,
        "--system-prompt", system_prompt,
        "--tools", "",
        "--setting-sources", "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--output-format", "json",
    ]


# Launched from inside a Claude Code session, the child would otherwise inherit that
# session's own settings (effort, base URL, entrypoint and about 30 more). Only what
# the CLI needs to run and to use the Claude Code sign-in is passed through. An API
# key is deliberately not passed, so calls never bill an API account.
_PASSED_ENV = {
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "LANG", "TERM",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
    "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR",
}


def child_environment(environ: Mapping[str, str]) -> dict[str, str]:
    return {k: v for k, v in environ.items() if k in _PASSED_ENV or k.startswith("LC_")}


def split_messages(messages: list[ChatMessage]) -> tuple[str, str]:
    system = [m.text for m in messages if isinstance(m, ChatMessageSystem)]
    rest = [m for m in messages if not isinstance(m, ChatMessageSystem)]
    if len(rest) != 1 or not isinstance(rest[0], ChatMessageUser):
        raise ClaudeCodeError("claude-code supports one user message plus optional system messages")
    return "\n\n".join(system) or DEFAULT_SYSTEM_PROMPT, rest[0].text


def parse_result(model: str, stdout: str) -> ModelOutput:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ClaudeCodeError(f"claude returned non-JSON output: {stdout[:200]!r}") from error
    if not isinstance(payload, dict):
        raise ClaudeCodeError(f"claude returned an unexpected shape: {stdout[:200]!r}")
    if payload.get("type", "result") != "result":
        raise ClaudeCodeError(f"claude returned a {payload.get('type')!r} message, not a result")
    if payload.get("is_error") or payload.get("subtype", "success") != "success":
        raise ClaudeCodeError(f"claude reported an error: {payload.get('subtype')} {payload.get('result')!r}")

    result = payload.get("result")
    if not isinstance(result, str) or not result.strip():
        raise ClaudeCodeError(f"claude returned no result: {stdout[:200]!r}")
    if result.strip().startswith(_IN_BAND_ERRORS):
        raise ClaudeCodeError(f"claude reported an error as its result: {result[:200]!r}")

    usage = payload.get("usage") or {}
    input_tokens = usage.get("input_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_write = usage.get("cache_creation_input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    output = ModelOutput.from_content(
        model=model,
        content=result,
        stop_reason=_STOP_REASONS.get(payload.get("stop_reason"), "stop"),
    )
    output.usage = ModelUsage(
        input_tokens=input_tokens,
        input_tokens_cache_read=cache_read,
        input_tokens_cache_write=cache_write,
        output_tokens=output_tokens,
        total_tokens=input_tokens + cache_read + cache_write + output_tokens,
    )
    return output


class ClaudeCodeAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: object,
    ) -> None:
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, config=config)

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        if tools:
            raise ClaudeCodeError("claude-code does not support tool calls")
        system_prompt, prompt = split_messages(input)
        binary = os.environ.get("CLAUDE_CODE_BIN", "claude")
        command = build_command(binary, self.model_name, system_prompt)
        timeout = config.timeout or DEFAULT_TIMEOUT_SECONDS

        # An empty working directory stops Claude Code finding a project CLAUDE.md.
        with tempfile.TemporaryDirectory() as workdir:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=workdir,
                env=child_environment(os.environ),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(prompt.encode()), timeout=timeout
                )
            except TimeoutError:
                raise ClaudeCodeError(f"claude did not answer within {timeout}s") from None
            finally:
                # Also runs on cancellation, so no claude process outlives its call.
                if process.returncode is None:
                    process.kill()
                    await process.wait()

        if process.returncode != 0:
            raise ClaudeCodeError(
                f"claude exited {process.returncode}: {stderr.decode(errors='replace')[-500:]}"
            )
        return parse_result(self.model_name, stdout.decode())
