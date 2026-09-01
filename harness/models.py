"""
Thin model-provider abstraction. Only Anthropic is wired up for this first pass; OpenAI and
local/HF providers are stubbed with TODOs so the harness can be extended without touching
run.py, scoring.py, or the task/payload data.

No API keys are hardcoded -- they are read from environment variables at call time.
"""

import os

DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
DEFAULT_MAX_TOKENS = 1024


def call_model(provider: str, model_name: str, prompt: str) -> str:
    """Generic entry point used by run.py: call_model(provider, model_name, prompt) -> raw text."""
    if provider == "anthropic":
        return _call_anthropic(model_name, prompt)
    if provider == "openai":
        return _call_openai(model_name, prompt)
    if provider in ("local", "hf", "huggingface"):
        return _call_local(model_name, prompt)
    raise ValueError(f"Unknown provider: {provider!r}")


def _call_anthropic(model_name: str, prompt: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it in your environment (do not commit it)."
        )

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model_name or DEFAULT_ANTHROPIC_MODEL,
        max_tokens=DEFAULT_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )


def _call_openai(model_name: str, prompt: str) -> str:
    # TODO(provider-expansion): wire up openai.OpenAI().chat.completions.create(...) here,
    # reading OPENAI_API_KEY from the environment. Return the response text as a plain string
    # to match _call_anthropic's contract.
    raise NotImplementedError("OpenAI provider is not wired up yet. See TODO in models.py.")


def _call_local(model_name: str, prompt: str) -> str:
    # TODO(provider-expansion): wire up a local/HF inference call here (e.g. transformers
    # pipeline or a self-hosted inference server). Return the response text as a plain string
    # to match _call_anthropic's contract.
    raise NotImplementedError("Local/HF provider is not wired up yet. See TODO in models.py.")
