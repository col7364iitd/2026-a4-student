# COL 764 Assignment 4 — Iterative Retrieval for Multi-hop QA (starter kit)

Read the handout first. This README is only about running the code.

You get a working **single-shot** QA system (baseline **B1**). It searches once with BM25 and asks the
staff-hosted LLM to answer from the top passages. Your job is to turn it into an **iterative** agent by
editing **`agent.py`**. You will write three functions:

| Function | What it decides |
|---|---|
| `next_query(question, notes, tools)` | what to search for next (e.g. the bridge entity) |
| `should_stop(question, notes, tools)` | whether the evidence is enough (or the budget is gone) |
| `answer(question, notes, tools)` | a short answer plus at most 3 cited `doc_id`s |

You don't touch indexing, the LLM server or evaluation. They are provided, and they behave exactly as in the
official harness.

---

## 1. Setup (about 10 minutes)

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt                          # numpy, requests
```

**Data.** Download the data bundle from <!-- TBD --> **[link on Moodle]** and unpack it so that you have:

```
data/corpus.jsonl     # ~139K paragraphs: {"doc_id", "title", "text"}
data/bm25/            # the prebuilt BM25 index (do not rebuild unless told to)
data/dev.jsonl        # ~300 dev questions with gold answers and supporting doc_ids
```

**LLM endpoint.** Copy the example config and fill in the values announced on Moodle:

```bash
cp config.example.json config.json
```

Check that everything works without using the server:

```bash
python run_dev.py --mock-llm --limit 5
```

You should see a results table, with all statuses `ok`. The mock LLM's answers are nonsense, so ignore the
scores. Then run the real baseline:

```bash
python run_dev.py --limit 50
```

## 2. Developing

```bash
python run_dev.py                                   # full dev set
python run_dev.py --limit 30 --workers 4            # quicker
python run_dev.py --ids 2hop__13548_13529           # specific questions
python run_dev.py --question "Who is the spouse of the Green performer?"   # one question + full trace
```

Each run writes `runs/<timestamp>/`:

* `metrics.json`: S, answer F1/EM, evidence F1, mean calls, status counts
* `review.jsonl`: one line per question, **worst first**: question, gold answer, your answer, cited vs. gold
  doc_ids. Start your error analysis here.
* `traces.jsonl`: every search query and LLM prompt/output, per question

Each line of `dev.jsonl` also has a `decomposition` field with the gold single-hop sub-questions and their
answers. Use it to see what the bridge entity should have been. It is **not** available at test time.

### Tools your agent can use

```python
tools.search(query, k=5)       -> list[Passage]   # BM25, 1 <= k <= 10
tools.llm(prompt, max_tokens)  -> str             # temperature 0, max_tokens clamped to 256
tools.budget_left()            -> {"search_calls": .., "llm_calls": .., "seconds": ..}
# Passage: .doc_id .title .text ; str(p) == "[doc_id] title: text"
```

### Budget per question (enforced)

| Resource | Limit |
|---|---|
| `tools.search` calls | 5 |
| `tools.llm` calls | 6 |
| prompt length per LLM call | 3,000 tokens: longer prompts raise `PromptTooLong` and are **not** counted |
| completion per LLM call | 256 tokens |
| wall-clock time | 60 s: then the question is killed and scores 0 |

Calls beyond the limit raise `BudgetExhausted` (import it from `a4`). Catch it and return your best answer.
The provided `run()` already does this.

### What you may import in your agent

Python standard library: `re json math string collections itertools functools operator dataclasses typing
heapq bisect unicodedata difflib statistics textwrap enum abc copy time random`, plus `numpy` and `a4`. Your
own helper modules can live in a `helpers/` folder next to `agent.py`.

Not allowed: file access (`open`), network, `os`, `sys`, `subprocess`, `eval`/`exec`, other models or
packages. The agent runs in a sandboxed process with no network access.

## 3. Submitting

**Leaderboard (dev set, at most 3 per day).**

```bash
python make_submission.py 2023CS10000 --no-report
```

Upload the zip at <!-- TBD --> **[leaderboard link on Moodle]**.

**Final (Moodle, by the deadline).**

```bash
python make_submission.py 2023CS10000     # agent.py + helpers/ + report.pdf, checked automatically
```

`check_submission.py <zip>` runs the same static checks as the official harness:

* only `.py` files and `report.pdf`
* code under 100 KB
* only allowed imports
* no large literal tables (these are reviewed)

Your last leaderboard submission must be identical to the `agent.py` you submit on Moodle.

## 4. Grading (summary; the handout is authoritative)

| Component | Weight |
|---|---|
| Functionality: runs cleanly, respects the budget, genuinely iterative | 20% |
| Leaderboard: hidden test set, S = 0.8 × answer F1 + 0.2 × evidence F1 | 45% |
| Report (≤ 2 pages) | 30% |
| Code quality | 5% |

Leaderboard marks depend on whether you beat the single-shot baseline B1 (score S_B1):

* **Below S_B1:** 0.4 × S/S_B1 of the marks.
* **At or above S_B1:** 0.5 + 0.5 × (your percentile among everyone else who also beat B1).

**Beating the single-shot baseline guarantees at least half of the leaderboard marks.**

## 5. FAQ

* **"Could not import your agent"**: there is a syntax error or a disallowed import. The message shows the traceback.
* **Status `timeout`**: one question took more than 60 s. Usually this means too many LLM calls with long
  outputs. Use a small `max_tokens` (e.g. 32–64) for short answers.
* **`PromptTooLong`**: pass fewer passages. `notes.as_text(4)` instead of `notes.as_text(8)`.
* **Do not edit `a4/`.** The official harness uses its own copy.
