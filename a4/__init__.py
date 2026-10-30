"""COL 764 Assignment 4 support package (shared by the student starter kit and the official harness).

Your agent may import from this package, e.g.

    from a4 import Passage, BudgetExhausted, PromptTooLong

Do NOT modify this package: the official harness uses its own trusted copy, so any change you make here
will not exist when your submission is graded.
"""

from .types import Passage
from .budget import (
    BUDGET,
    Budget,
    BudgetExhausted,
    PromptTooLong,
    LLMError,
)

__version__ = "1.0.0"

__all__ = [
    "Passage",
    "BUDGET",
    "Budget",
    "BudgetExhausted",
    "PromptTooLong",
    "LLMError",
    "__version__",
]
