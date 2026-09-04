"""
Thin model-provider abstraction covering the five models evaluated in the paper (Table:
"Overview of Evaluated Language Models"): GPT-4o, Claude 4.6 Sonnet, Meta-Llama-3.1-8B-Instruct,
Mistral-7B, and Qwen2.5-7B-Instruct.

No API keys are hardcoded -- they are read from environment variables at call time.
"""

import os

DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
DEFAULT_MAX_TOKENS = 1024

# Friendly name (as used in results.csv and the paper's model table) -> provider + the exact
# model identifier passed to that provider's API. The three open-weight models are called
# through the Hugging Face Inference API rather than run locally, since they are 7-8B instruct
# checkpoints that would otherwise require local GPU inference. Override any of the HF repo IDs
# via env vars if a provider's hosted route for a given checkpoint changes.
MODELS = {
    "gpt-4o": {
        "provider": "openai",
        "model_name": os.environ.get("OPENAI_MODEL", "gpt-4o"),
    },
    "claude-sonnet-5": {
        "provider": "anthropic",
        "model_name": DEFAULT_ANTHROPIC_MODEL,
    },
    "Meta-Llama-3.1-8B-Instruct": {
        "provider": "hf",
        "model_name": os.environ.get("HF_LLAMA_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    },
    "Mistral-7B": {
        "provider": "hf",
        "model_name": os.environ.get("HF_MISTRAL_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
    },
    "Qwen2.5-7B-Instruct": {
        "provider": "hf",
        "model_name": os.environ.get("HF_QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    },
}


def call_model(provider: str, model_name: str, prompt: str) -> str:
    """Generic entry point used by run.py: call_model(provider, model_name, prompt) -> raw text."""
    if provider == "anthropic":
        return _call_anthropic(model_name, prompt)
    if provider == "openai":
        return _call_openai(model_name, prompt)
    if provider in ("hf", "huggingface"):
        return _call_hf(model_name, prompt)
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
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it in your environment (do not commit it)."
        )

    import openai

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        max_tokens=DEFAULT_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def _call_hf(model_name: str, prompt: str) -> str:
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        raise RuntimeError(
            "HF_TOKEN is not set. Export it in your environment (do not commit it)."
        )

    from huggingface_hub import InferenceClient

    client = InferenceClient(model=model_name, token=api_key)
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    return response.choices[0].message.content or ""
