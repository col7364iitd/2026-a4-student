"""A small, deterministic BM25 index in pure Python + NumPy.

Students do not need to change or tune anything here: the index and its parameters are fixed by staff,
and the official harness uses exactly this implementation.

Scoring (Lucene-style BM25, Robertson & Zaragoza 2009):

    score(q, d) = sum_{t in q} idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

Query terms are de-duplicated. Ties are broken by corpus order, so results are fully deterministic.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from array import array
from collections import Counter
from typing import Iterable

import numpy as np

K1 = 0.9
B = 0.4

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his i if in into is it its of on or
    she that the their them they this to was were which who whom will with you your""".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case, strip accents, split on non-alphanumerics, drop a short stopword list."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return [t for t in _TOKEN_RE.findall(text) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, vocab, offsets, post_docs, post_tf, doc_len, k1=K1, b=B):
        self.vocab: dict[str, int] = vocab
        self.offsets = offsets          # int64, len = |V| + 1
        self.post_docs = post_docs      # int32, doc indices grouped by term
        self.post_tf = post_tf          # int32, term frequencies aligned with post_docs
        self.doc_len = doc_len          # float32, len = N
        self.k1 = k1
        self.b = b
        self.n_docs = int(doc_len.shape[0])
        self.avgdl = float(doc_len.mean()) if self.n_docs else 0.0
        df = np.diff(offsets).astype(np.float64)
        self.idf = np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        self._norm = (self.k1 * (1.0 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))).astype(np.float32)

    # ------------------------------------------------------------------ build / io
    @classmethod
    def build(cls, docs: Iterable[str], k1=K1, b=B) -> "BM25Index":
        """``docs`` yields one string per document (title + text), in corpus order."""
        vocab: dict[str, int] = {}
        term_ids = array("i")
        doc_ids = array("i")
        tfs = array("i")
        doc_len = array("f")
        for d, text in enumerate(docs):
            toks = tokenize(text)
            doc_len.append(len(toks))
            for tok, c in Counter(toks).items():
                tid = vocab.setdefault(tok, len(vocab))
                term_ids.append(tid)
                doc_ids.append(d)
                tfs.append(c)
        t = np.frombuffer(term_ids, dtype=np.int32)
        dd = np.frombuffer(doc_ids, dtype=np.int32)
        tf = np.frombuffer(tfs, dtype=np.int32)
        order = np.lexsort((dd, t))  # by term, then doc
        t, dd, tf = t[order], dd[order], tf[order]
        counts = np.bincount(t, minlength=len(vocab))
        offsets = np.zeros(len(vocab) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        return cls(vocab, offsets, dd.copy(), tf.copy(), np.frombuffer(doc_len, dtype=np.float32).copy(), k1, b)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        np.savez(
            os.path.join(directory, "postings.npz"),
            offsets=self.offsets,
            post_docs=self.post_docs,
            post_tf=self.post_tf,
            doc_len=self.doc_len,
        )
        terms = [None] * len(self.vocab)
        for tok, i in self.vocab.items():
            terms[i] = tok
        with open(os.path.join(directory, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump({"k1": self.k1, "b": self.b, "terms": terms}, f)

    @classmethod
    def load(cls, directory: str) -> "BM25Index":
        with open(os.path.join(directory, "vocab.json"), encoding="utf-8") as f:
            meta = json.load(f)
        z = np.load(os.path.join(directory, "postings.npz"))
        vocab = {tok: i for i, tok in enumerate(meta["terms"])}
        return cls(vocab, z["offsets"], z["post_docs"], z["post_tf"], z["doc_len"], meta["k1"], meta["b"])

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Return up to ``k`` (doc_index, score) pairs, best first."""
        tids = sorted({self.vocab[t] for t in tokenize(query) if t in self.vocab})
        if not tids or k <= 0:
            return []
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for tid in tids:
            s, e = self.offsets[tid], self.offsets[tid + 1]
            docs = self.post_docs[s:e]
            tf = self.post_tf[s:e].astype(np.float32)
            scores[docs] += self.idf[tid] * tf * (self.k1 + 1.0) / (tf + self._norm[docs])
        cand = np.flatnonzero(scores > 0)
        if cand.size == 0:
            return []
        order = np.lexsort((cand, -scores[cand]))[:k]  # score desc, then corpus order
        top = cand[order]
        return [(int(i), float(scores[i])) for i in top]
