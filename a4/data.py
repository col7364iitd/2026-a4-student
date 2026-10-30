"""Loading the corpus, the question files and the BM25 index."""

from __future__ import annotations

import json
import os

from .bm25 import BM25Index
from .types import Passage


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class Corpus:
    """The pooled MuSiQue paragraph corpus. Each line of corpus.jsonl: {doc_id, title, text}."""

    def __init__(self, path: str):
        self.doc_ids: list[str] = []
        self.titles: list[str] = []
        self.texts: list[str] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                self.doc_ids.append(r["doc_id"])
                self.titles.append(r["title"])
                self.texts.append(r["text"])
        self.index_of = {d: i for i, d in enumerate(self.doc_ids)}

    def __len__(self) -> int:
        return len(self.doc_ids)

    def passage(self, i: int, score: float = 0.0) -> Passage:
        return Passage(self.doc_ids[i], self.titles[i], self.texts[i], score)

    def index_text(self, i: int) -> str:
        """The text that BM25 indexes for document i: title followed by paragraph."""
        return self.titles[i] + " " + self.texts[i]


class SearchEngine:
    """Corpus + BM25 index, returning Passage objects."""

    def __init__(self, corpus: Corpus, index: BM25Index):
        if index.n_docs != len(corpus):
            raise ValueError(f"Index has {index.n_docs} docs but corpus has {len(corpus)}; rebuild the index.")
        self.corpus = corpus
        self.index = index

    @classmethod
    def load(cls, data_dir: str) -> "SearchEngine":
        corpus = Corpus(os.path.join(data_dir, "corpus.jsonl"))
        index = BM25Index.load(os.path.join(data_dir, "bm25"))
        return cls(corpus, index)

    def search(self, query: str, k: int) -> list[Passage]:
        return [self.corpus.passage(i, s) for i, s in self.index.search(query, k)]
