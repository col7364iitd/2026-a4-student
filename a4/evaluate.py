"""Scoring, exactly as used for the leaderboard (handout Section 4.1).

* Answers are normalised as in the official SQuAD v1.1 evaluation script (Rajpurkar et al., EMNLP 2016):
  lower-case, remove punctuation, remove the articles a/an/the, collapse whitespace.
* Answer F1 / EM: best match over the gold answer and all its aliases.
* Evidence F1: F1 between the first 3 distinct cited doc_ids and the gold supporting doc_ids.
* Leaderboard score: S = 0.8 * mean answer F1 + 0.2 * mean evidence F1.
* A question with status other than "ok" (crash / timeout / malformed), or with no prediction, scores 0.
"""

from __future__ import annotations

import re
import string
from collections import Counter

from .budget import BUDGET

W_ANSWER = 0.8
W_EVIDENCE = 0.2

_PUNCT = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def answer_scores(pred: str, golds: list[str]) -> tuple[float, float]:
    """Returns (exact_match, f1), each maximised over the gold answers."""
    golds = [g for g in golds if g is not None]
    em = max(float(normalize_answer(pred) == normalize_answer(g)) for g in golds)
    f1 = max(_f1(pred, g) for g in golds)
    return em, f1


def evidence_f1(cited: list[str], gold_ids: list[str], max_cit: int = BUDGET.max_citations_scored) -> float:
    seen, cits = set(), []
    for c in cited:
        if c not in seen:
            seen.add(c)
            cits.append(c)
        if len(cits) == max_cit:
            break
    gold = set(gold_ids)
    if not cits or not gold:
        return 0.0
    tp = len(set(cits) & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(cits), tp / len(gold)
    return 2 * p * r / (p + r)


def score(records: list[dict], gold_rows: list[dict]) -> dict:
    """Score prediction records (from the runner) against gold rows (dev.jsonl / test.jsonl format)."""
    by_id = {r["id"]: r for r in records}
    per_q = []
    status_counts: Counter = Counter()
    n_search = n_llm = 0
    for g in gold_rows:
        r = by_id.get(g["id"])
        status = r["status"] if r else "missing"
        status_counts[status] += 1
        if r:
            n_search += r.get("n_search", 0)
            n_llm += r.get("n_llm", 0)
        if status == "ok":
            em, f1 = answer_scores(r["answer"], [g["answer"]] + list(g.get("answer_aliases", [])))
            ev = evidence_f1(r["citations"], g["supporting_doc_ids"])
        else:
            em = f1 = ev = 0.0
        per_q.append({"id": g["id"], "status": status, "em": em, "f1": f1, "ev_f1": ev})
    n = max(1, len(gold_rows))
    em = sum(q["em"] for q in per_q) / n
    f1 = sum(q["f1"] for q in per_q) / n
    ev = sum(q["ev_f1"] for q in per_q) / n
    return {
        "n": len(gold_rows),
        "S": round(W_ANSWER * f1 + W_EVIDENCE * ev, 6),
        "answer_f1": round(f1, 6),
        "answer_em": round(em, 6),
        "evidence_f1": round(ev, 6),
        "mean_search_calls": round(n_search / n, 3),
        "mean_llm_calls": round(n_llm / n, 3),
        "status_counts": dict(status_counts),
        "per_question": per_q,
    }
