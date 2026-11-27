from inspect_ai.model import modelapi


@modelapi(name="claude-code")
def claude_code():
    from claude_code_llm.provider import ClaudeCodeAPI

    return ClaudeCodeAPI
