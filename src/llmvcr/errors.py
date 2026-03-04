"""Custom exceptions for llmvcr."""


class LLMVCRError(Exception):
    """Base exception for all llmvcr errors."""


class CassetteNotFoundError(LLMVCRError):
    """Raised when playback mode is used but no cassette file exists."""

    def __init__(self, path: str):
        self.path = path
        super().__init__(
            f"Cassette not found: '{path}'\n"
            f"Run once in record mode to create it:\n"
            f"  @llmvcr.cassette('{path}', mode='record')"
        )


class NoMatchFoundError(LLMVCRError):
    """Raised when a request has no matching interaction in the cassette."""

    def __init__(self, path: str, request_summary: str):
        self.path = path
        self.request_summary = request_summary
        super().__init__(
            f"No matching interaction found in cassette '{path}' for request:\n"
            f"  {request_summary}\n"
            f"Re-record the cassette or add the interaction manually."
        )


class ProviderNotSupportedError(LLMVCRError):
    """Raised when llmvcr cannot identify the LLM provider being patched."""

    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(
            f"Provider '{provider}' is not supported yet.\n"
            f"Supported providers: openai, anthropic\n"
            f"Open an issue at https://github.com/your-username/llmvcr/issues"
        )