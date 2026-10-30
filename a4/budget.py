"""Per-question budget for Assignment 4 and the exceptions the tools can raise.

The official limits are the ones in ``BUDGET`` below. The official harness uses its own copy of this file,
so editing it changes nothing about how you are graded.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Budget:
    max_search_calls: int = 5          # tools.search calls per question
    max_llm_calls: int = 6             # tools.llm calls per question
    max_prompt_tokens: int = 3000      # per LLM call; longer prompts raise PromptTooLong (not counted)
    max_completion_tokens: int = 256   # per LLM call; larger max_tokens values are clamped to this
    max_k: int = 10                    # largest k allowed in tools.search
    wall_clock_seconds: float = 60.0   # per question, enforced by the runner
    max_citations_scored: int = 3      # only the first 3 cited doc_ids are scored


BUDGET = Budget()


class BudgetExhausted(Exception):
    """Raised when a search or LLM call would exceed the per-question budget.

    Catch it in your agent and return your best answer with what you have.
    """


class PromptTooLong(Exception):
    """Raised when a prompt is longer than ``BUDGET.max_prompt_tokens`` tokens.

    The call is *not* counted against your budget; shorten the prompt and try again.
    """


class LLMError(Exception):
    """The LLM server could not be reached or returned an error after several retries.

    In official runs, questions affected by server-side errors are re-run by staff.
    """
