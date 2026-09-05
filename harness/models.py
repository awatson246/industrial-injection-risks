"""
Thin model-provider abstraction covering the five models evaluated in the paper (Table:
"Overview of Evaluated Language Models"): GPT-4o, Claude 4.6 Sonnet, Meta-Llama-3.1-8B-Instruct,
Mistral-7B, and Qwen2.5-7B-Instruct.

No API keys are hardcoded -- they are read from environment variables at call time.
"""

import os

import requests

DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
DEFAULT_MAX_TOKENS = 1024

# Friendly name (as used in results.csv and the paper's model table) -> provider + the exact
# model identifier passed to that provider's API. Llama and Qwen are called through the Hugging
# Face Inference Providers marketplace (7-8B instruct checkpoints that would otherwise require
# local GPU inference). Mistral-7B is called directly against Mistral AI's own API instead --
# no current HF Inference Providers route serves a Mistral-7B-Instruct checkpoint with chat
# support, but Mistral's own platform hosts the original open 7B checkpoint as "open-mistral-7b".
# Override any of the hosted repo IDs via env vars if a provider's route for a checkpoint changes.
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
        "provider": "mistral",
        "model_name": os.environ.get("MISTRAL_MODEL", "open-mistral-7b"),
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
    if provider == "mistral":
        return _call_mistral(model_name, prompt)
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


def _call_mistral(model_name: str, prompt: str) -> str:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is not set. Export it in your environment (do not commit it)."
        )

    # Called directly against Mistral AI's own API (OpenAI-compatible REST, no SDK dependency)
    # rather than through Hugging Face -- see the MODELS registry comment above for why.
    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": DEFAULT_MAX_TOKENS,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"] or ""
