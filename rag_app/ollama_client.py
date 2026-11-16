"""Ollama client (chat + embeddings) plus a deterministic fake for tests.

Real client talks to a local Ollama server over its HTTP API using httpx.
Configuration comes from the environment:

- ``OLLAMA_HOST``        (default ``http://localhost:11434``)
- ``OLLAMA_MODEL``       chat model      (default ``llama3.1:8b``)
- ``OLLAMA_EMBED_MODEL`` embedding model (default ``nomic-embed-text``)

``FakeOllamaClient`` produces deterministic bag-of-words embeddings from a
hashed token projection, so tests and the offline eval run without Ollama
or any network access. Its ``chat`` simply echoes the first supplied
passage, which is enough to exercise the /ask flow end to end.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Iterable, Sequence

import httpx

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_CHAT_MODEL = "llama3.1:8b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Very common words carry no topical signal; dropping them keeps the fake
# embeddings (and only the fake ones) focused on content words.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was",
    "were", "be", "been", "it", "its", "this", "that", "these", "those",
    "on", "for", "with", "as", "by", "at", "from", "how", "what", "why",
    "when", "who", "do", "does", "did", "can", "could", "should", "would",
    "will", "you", "your", "they", "them", "their", "if", "then", "than",
    "so", "such", "into", "about", "over", "under", "not", "no", "yes",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class OllamaClient:
    """Thin wrapper over the Ollama HTTP API."""

    def __init__(
        self,
        host: str | None = None,
        chat_model: str | None = None,
        embed_model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.host = (host or os.getenv("OLLAMA_HOST", DEFAULT_HOST)).rstrip("/")
        self.chat_model = chat_model or os.getenv("OLLAMA_MODEL", DEFAULT_CHAT_MODEL)
        self.embed_model = embed_model or os.getenv(
            "OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL
        )
        self.timeout = timeout

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts; returns one vector per input text."""
        if not texts:
            return []
        with httpx.Client(base_url=self.host, timeout=self.timeout) as client:
            resp = client.post(
                "/api/embed",
                json={"model": self.embed_model, "input": list(texts)},
            )
            resp.raise_for_status()
            data = resp.json()
        return data["embeddings"]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def chat(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        with httpx.Client(base_url=self.host, timeout=self.timeout) as client:
            resp = client.post(
                "/api/chat",
                json={
                    "model": self.chat_model,
                    "messages": messages,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        return data["message"]["content"]


class FakeOllamaClient:
    """Deterministic, offline stand-in with the same interface.

    Embeddings: each content token is hashed into a fixed-size vector
    (signed hashing trick), counts are accumulated and the result is
    L2-normalised. Texts that share content words get high cosine
    similarity, which is all the retrieval tests need.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self.chat_model = "fake-chat"
        self.embed_model = "fake-embed"

    def _embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._embed_text(text)

    def chat(self, prompt: str, system: str | None = None) -> str:
        # Echo the first [1] passage so API tests can assert the plumbing
        # carried retrieved context into the answer.
        for line in prompt.splitlines():
            if line.startswith("[1]"):
                return f"Based on the guidance: {line[3:].strip()}"
        return "Based on the guidance provided."


def make_client(fake: bool = False) -> OllamaClient | FakeOllamaClient:
    """Return the real Ollama client, or the fake when ``fake`` is set."""
    if fake or os.getenv("RAG_FAKE_LLM") == "1":
        return FakeOllamaClient()
    return OllamaClient()
