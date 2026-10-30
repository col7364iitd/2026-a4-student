"""The ``tools`` object your agent receives, and its budget accounting.

API available to your agent (see the handout, Section 2.2):

    tools.search(query: str, k: int = 5) -> list[Passage]   # BM25 over the corpus, 1 <= k <= 10
    tools.llm(prompt: str, max_tokens: int = 256) -> str     # staff-hosted LLM, temperature 0
    tools.budget_left() -> dict                              # remaining calls for this question

In the runner, these calls are forwarded from your agent's process to a trusted parent process,
which owns the search engine, the LLM connection and the budget. This module implements that trusted side
(``Tools``) and the thin proxy that your agent actually holds (``ToolsProxy``).
"""

from __future__ import annotations

import time

from .budget import BUDGET, Budget, BudgetExhausted, LLMError, PromptTooLong
from .types import Passage


class Tools:
    """Trusted, per-question tool implementation with budget enforcement and a call trace."""

    def __init__(self, engine, llm, budget: Budget = BUDGET):
        self._engine = engine
        self._llm = llm
        self._budget = budget
        self._n_search = 0
        self._n_llm = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._start = time.monotonic()
        self.trace: list[dict] = []
        self.infra_error = False

    # -- API -------------------------------------------------------------------------------------------
    def search(self, query: str, k: int = 5) -> list[Passage]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        k = int(k)
        if not 1 <= k <= self._budget.max_k:
            raise ValueError(f"k must be between 1 and {self._budget.max_k}")
        if self._n_search >= self._budget.max_search_calls:
            raise BudgetExhausted(f"search budget of {self._budget.max_search_calls} calls used up")
        self._n_search += 1
        results = self._engine.search(query, k)
        self.trace.append({"type": "search", "query": query, "k": k, "doc_ids": [p.doc_id for p in results]})
        return results

    def llm(self, prompt: str, max_tokens: int = 256) -> str:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if self._n_llm >= self._budget.max_llm_calls:
            raise BudgetExhausted(f"LLM budget of {self._budget.max_llm_calls} calls used up")
        max_tokens = max(1, min(int(max_tokens), self._budget.max_completion_tokens))
        try:
            n_prompt = self._llm.count_tokens(prompt)
        except LLMError:
            self.infra_error = True
            raise
        if n_prompt > self._budget.max_prompt_tokens:
            self.trace.append({"type": "llm_rejected", "prompt_tokens": n_prompt})
            raise PromptTooLong(
                f"prompt has {n_prompt} tokens; the limit is {self._budget.max_prompt_tokens} (call not counted)"
            )
        self._n_llm += 1
        try:
            text, p_tok, c_tok = self._llm.complete(prompt, max_tokens)
        except LLMError:
            self.infra_error = True
            raise
        self._prompt_tokens += p_tok or n_prompt
        self._completion_tokens += c_tok
        self.trace.append(
            {"type": "llm", "prompt": prompt, "output": text, "prompt_tokens": p_tok or n_prompt,
             "completion_tokens": c_tok, "max_tokens": max_tokens}
        )
        return text

    def budget_left(self) -> dict:
        return {
            "search_calls": self._budget.max_search_calls - self._n_search,
            "llm_calls": self._budget.max_llm_calls - self._n_llm,
            "seconds": round(max(0.0, self._budget.wall_clock_seconds - (time.monotonic() - self._start)), 2),
        }

    # -- bookkeeping (not part of the agent API) -------------------------------------------------------
    def usage(self) -> dict:
        return {
            "n_search": self._n_search,
            "n_llm": self._n_llm,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
        }


_EXC = {
    "BudgetExhausted": BudgetExhausted,
    "PromptTooLong": PromptTooLong,
    "LLMError": LLMError,
    "ValueError": ValueError,
    "TypeError": TypeError,
}

_ALLOWED_METHODS = ("search", "llm", "budget_left")


class ToolsProxy:
    """What your agent receives inside the sandboxed worker process.

    Every call is sent to the trusted parent process over a pipe; the parent enforces the budget.
    """

    __slots__ = ("_conn",)

    def __init__(self, conn):
        object.__setattr__(self, "_conn", conn)

    def _call(self, method, *args, **kwargs):
        self._conn.send(("call", method, args, kwargs))
        kind, payload = self._conn.recv()
        if kind == "ok":
            return payload
        exc_name, msg = payload
        raise _EXC.get(exc_name, RuntimeError)(msg)

    def search(self, query: str, k: int = 5) -> list[Passage]:
        return self._call("search", query, k)

    def llm(self, prompt: str, max_tokens: int = 256) -> str:
        return self._call("llm", prompt, max_tokens)

    def budget_left(self) -> dict:
        return self._call("budget_left")

    def __setattr__(self, name, value):
        raise AttributeError("tools is read-only")


def dispatch(tools: Tools, method: str, args, kwargs):
    """Execute one proxied call on the trusted side; returns a ('ok'|'err', payload) message."""
    if method not in _ALLOWED_METHODS:
        return ("err", ("ValueError", f"unknown tool method {method!r}"))
    try:
        return ("ok", getattr(tools, method)(*args, **kwargs))
    except tuple(_EXC.values()) as e:
        return ("err", (type(e).__name__, str(e)))
    except Exception as e:  # anything unexpected becomes a RuntimeError in the agent
        return ("err", ("RuntimeError", f"{type(e).__name__}: {e}"))
