"""LLM backends used by the tools.

* ``HTTPLLM``  - the staff-hosted model behind an OpenAI-compatible server (vLLM). Used in official runs.
* ``MockLLM``  - a tiny offline stand-in, so you can test plumbing without the server (``--mock-llm``).
  Its answers are meaningless; never report numbers obtained with it.

Both expose:
    count_tokens(prompt) -> int
    complete(prompt, max_tokens) -> (text, prompt_tokens, completion_tokens)
"""

from __future__ import annotations

import math
import re
import time

from .budget import LLMError
from .config import LLMConfig


class HTTPLLM:
    def __init__(self, cfg: LLMConfig, retries: int = 3):
        if not cfg.llm_base_url or not cfg.llm_model:
            raise ValueError(
                "LLM endpoint not configured: set llm_base_url and llm_model in config.json "
                "(see config.example.json), or run with --mock-llm."
            )
        import requests  # imported here so the agent process never needs it

        self._requests = requests
        self.cfg = cfg
        self.retries = retries
        self.headers = {"Content-Type": "application/json"}
        if cfg.llm_api_key:
            self.headers["Authorization"] = f"Bearer {cfg.llm_api_key}"

    def _post(self, path: str, payload: dict) -> dict:
        url = self.cfg.llm_base_url + path
        last = None
        for attempt in range(self.retries):
            try:
                r = self._requests.post(url, json=payload, headers=self.headers, timeout=self.cfg.request_timeout)
                if r.status_code < 500:
                    r.raise_for_status()
                    return r.json()
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            except self._requests.exceptions.HTTPError as e:
                raise LLMError(f"LLM request rejected: {e}; body: {r.text[:300]}") from e
            except Exception as e:  # connection errors, timeouts
                last = repr(e)
            time.sleep(2 ** attempt)
        raise LLMError(f"LLM server error after {self.retries} attempts: {last}")

    def count_tokens(self, prompt: str) -> int:
        if self.cfg.token_count == "approx":
            return approx_tokens(prompt)
        out = self._post("/tokenize", {"model": self.cfg.llm_model, "prompt": prompt, "add_special_tokens": False})
        if "count" in out:
            return int(out["count"])
        return len(out["tokens"])

    def complete(self, prompt: str, max_tokens: int) -> tuple[str, int, int]:
        payload = {
            "model": self.cfg.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "max_tokens": int(max_tokens),
        }
        payload.update(self.cfg.extra_body or {})
        out = self._post("/v1/chat/completions", payload)
        text = out["choices"][0]["message"].get("content") or ""
        usage = out.get("usage") or {}
        return text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


def approx_tokens(text: str) -> int:
    """Rough token estimate (about 4 characters per token). Used only by the mock backend."""
    return max(1, math.ceil(len(text) / 4))


_PASSAGE_LINE = re.compile(r"^\[(?P<doc>[^\]]+)\]\s*(?P<title>[^:\n]+):", re.MULTILINE)


class MockLLM:
    """Deterministic fake model for offline testing.

    It returns the title of the first passage it finds in the prompt (lines formatted as
    ``[doc_id] Title: text``, which is what ``str(Passage)`` produces), or the first few words of the
    prompt otherwise. This is enough to exercise your loop, budgets and output format - nothing more.
    """

    def count_tokens(self, prompt: str) -> int:
        return approx_tokens(prompt)

    def complete(self, prompt: str, max_tokens: int) -> tuple[str, int, int]:
        m = _PASSAGE_LINE.search(prompt)
        text = m.group("title").strip() if m else " ".join(prompt.split()[:5])
        words = text.split()[: max(1, max_tokens)]
        text = " ".join(words)
        return text, approx_tokens(prompt), approx_tokens(text)
