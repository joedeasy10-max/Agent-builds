"""Answer synthesis behind one thin interface. The provider is interchangeable.

Providers:
  * echo      - deterministic, offline, zero-cost. Stitches an answer from the
                retrieved context (or refuses when there is no context). Used by
                tests and any offline run, so the judge pipeline is exercisable
                without a key or network.
  * openai    - an OpenAI chat model (default gpt-4o-mini). Needs OPENAI_API_KEY.
  * anthropic - a Claude model (default claude-sonnet-5). Needs ANTHROPIC_API_KEY.

Both real providers answer strictly from the supplied context with the same
system instruction, so swapping `generation.provider` is a one-line config
change (leave `generation.model` blank to take the provider's default). Note
Anthropic has no embeddings API — an Anthropic-only setup pairs Claude here with
a local `bge` embedder (see src/embed.py); only generation/judge are Claude.
"""

from __future__ import annotations

import re
from typing import Protocol

from .config import Config, GenerationConfig

_REFUSAL = "I don't have enough information in the provided guidance to answer that."
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"

# Shared instruction so the two real providers behave the same way.
_SYSTEM = (
    "You answer questions using ONLY the provided GOV.UK context. "
    "If the context does not contain the answer, say you don't have enough "
    "information. Be concise."
)


def _prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts))
    return f"Context:\n{context_block}\n\nQuestion: {question}"


class Generator(Protocol):
    generator_id: str

    def generate(self, question: str, contexts: list[str]) -> str:
        ...


class EchoGenerator:
    """Deterministic, offline. Answers from the top context; refuses if none."""

    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences
        self.generator_id = "echo"

    def generate(self, question: str, contexts: list[str]) -> str:
        joined = " ".join(c.strip() for c in contexts if c.strip())
        if not joined:
            return _REFUSAL
        sentences = _SENT_RE.split(joined)
        return " ".join(sentences[: self.max_sentences]).strip()


class OpenAIGenerator:
    """An OpenAI chat model. Answers strictly from the supplied context."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 512):
        self.model = model or _DEFAULT_OPENAI_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.generator_id = f"openai-{self.model}"

    def generate(self, question: str, contexts: list[str]) -> str:
        from openai import OpenAI  # lazy: only on the real path

        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _prompt(question, contexts)},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


class AnthropicGenerator:
    """A Claude model. Answers strictly from the supplied context."""

    def __init__(self, model: str, max_tokens: int = 512):
        # A model configured for the other provider (e.g. a gpt-* id) falls back
        # to the Anthropic default, so flipping only `provider` still works.
        self.model = model if model.startswith("claude") else _DEFAULT_ANTHROPIC_MODEL
        self.max_tokens = max_tokens
        self.generator_id = f"anthropic-{self.model}"

    def generate(self, question: str, contexts: list[str]) -> str:
        import anthropic  # lazy: only on the real path

        client = anthropic.Anthropic()
        # No temperature/thinking params: newer Claude models reject sampling
        # params, and the default (adaptive) behaviour is what we want here.
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM,
            messages=[{"role": "user", "content": _prompt(question, contexts)}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()


def build_generator(config: Config) -> Generator:
    gen: GenerationConfig = config.generation
    if gen.provider == "echo":
        return EchoGenerator()
    if gen.provider == "openai":
        return OpenAIGenerator(
            model=gen.model, temperature=gen.temperature, max_tokens=gen.max_tokens
        )
    if gen.provider == "anthropic":
        return AnthropicGenerator(model=gen.model, max_tokens=gen.max_tokens)
    raise ValueError(f"Unknown generation.provider: {gen.provider!r}")
