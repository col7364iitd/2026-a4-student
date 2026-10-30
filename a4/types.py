from dataclasses import dataclass


@dataclass(frozen=True)
class Passage:
    """One paragraph of the corpus, as returned by ``tools.search``."""

    doc_id: str
    title: str
    text: str
    score: float = 0.0

    def __str__(self) -> str:  # handy when building prompts
        return f"[{self.doc_id}] {self.title}: {self.text}"
