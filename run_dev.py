#!/usr/bin/env python3
"""Run your agent on the dev set (or a single question) and score it - the same way the harness does.

Examples
--------
  python run_dev.py                          # full dev set with the staff LLM (needs config.json)
  python run_dev.py --limit 20               # first 20 dev questions
  python run_dev.py --mock-llm --limit 5     # offline plumbing test (scores are meaningless)
  python run_dev.py --question "Who founded the company that makes the iPhone?"   # one question + trace

Outputs go to runs/<timestamp>/:
  metrics.json        overall scores (S, answer F1, evidence F1, EM, mean calls, status counts)
  predictions.jsonl   your answers and citations, status, calls used
  traces.jsonl        every search / LLM call (queries, doc_ids, prompts, outputs) - use for error analysis
  review.jsonl        one line per question: question, gold, your answer, scores (sorted worst first)
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a4.config import load_config  # noqa: E402
from a4.data import SearchEngine, read_jsonl, write_jsonl  # noqa: E402
from a4.evaluate import score  # noqa: E402
from a4.llm import HTTPLLM, MockLLM  # noqa: E402
from a4.runner import AgentImportError, run_agent  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="agent.py")
    ap.add_argument("--data-dir", default="data", help="folder with corpus.jsonl, bm25/ and dev.jsonl")
    ap.add_argument("--split", default="dev.jsonl", help="question file inside --data-dir")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ap.add_argument("--ids", default="", help="comma-separated question ids to run")
    ap.add_argument("--question", default="", help="run a single ad-hoc question and print its trace")
    ap.add_argument("--workers", type=int, default=8, help="questions answered in parallel")
    ap.add_argument("--mock-llm", action="store_true", help="offline fake LLM (for testing plumbing only)")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out", default="", help="output folder (default runs/<timestamp>)")
    args = ap.parse_args()

    print("Loading corpus and BM25 index ...", file=sys.stderr)
    engine = SearchEngine.load(args.data_dir)
    llm = MockLLM() if args.mock_llm else HTTPLLM(load_config(args.config))
    if args.mock_llm:
        print("WARNING: using the mock LLM - scores are meaningless.", file=sys.stderr)

    if args.question:
        recs, traces = run_agent(args.agent, [{"id": "adhoc", "question": args.question}], engine, llm,
                                 workers=1, progress=False)
        print(json.dumps(traces[0], indent=2, ensure_ascii=False))
        r = recs[0]
        print(f"\nstatus={r['status']}  answer={r['answer']!r}  citations={r['citations']}")
        print(f"calls: search={r['n_search']} llm={r['n_llm']}  time={r['elapsed']}s")
        if r["error"]:
            print("\nERROR:\n" + r["error"])
        return

    gold = read_jsonl(os.path.join(args.data_dir, args.split))
    if args.ids:
        wanted = set(args.ids.split(","))
        gold = [g for g in gold if g["id"] in wanted]
    if args.limit:
        gold = gold[: args.limit]
    questions = [{"id": g["id"], "question": g["question"]} for g in gold]

    print(f"Running {args.agent} on {len(questions)} questions with {args.workers} workers ...", file=sys.stderr)
    t0 = time.time()
    try:
        recs, traces = run_agent(args.agent, questions, engine, llm, workers=args.workers)
    except AgentImportError as e:
        sys.exit(f"Could not import your agent:\n{e}")
    m = score(recs, gold)

    out = args.out or os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    write_jsonl(os.path.join(out, "predictions.jsonl"), recs)
    write_jsonl(os.path.join(out, "traces.jsonl"), traces)
    rec_by_id = {r["id"]: r for r in recs}
    review = []
    for g, q in zip(gold, m["per_question"]):
        r = rec_by_id[g["id"]]
        review.append({"id": g["id"], "question": g["question"], "gold": g["answer"],
                       "aliases": g.get("answer_aliases", []), "pred": r["answer"], "status": r["status"],
                       "f1": round(q["f1"], 3), "ev_f1": round(q["ev_f1"], 3), "citations": r["citations"],
                       "gold_doc_ids": g["supporting_doc_ids"], "n_search": r["n_search"], "n_llm": r["n_llm"],
                       "error": r["error"][-500:]})
    review.sort(key=lambda x: (x["f1"], x["ev_f1"]))
    write_jsonl(os.path.join(out, "review.jsonl"), review)
    summary = {k: v for k, v in m.items() if k != "per_question"}
    summary.update({"agent": args.agent, "mock_llm": args.mock_llm, "seconds": round(time.time() - t0, 1)})
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ results ================")
    print(f"questions          {m['n']}")
    print(f"S (leaderboard)    {m['S']:.4f}   = 0.8 x answer F1 + 0.2 x evidence F1")
    print(f"answer F1          {m['answer_f1']:.4f}")
    print(f"answer EM          {m['answer_em']:.4f}")
    print(f"evidence F1        {m['evidence_f1']:.4f}")
    print(f"mean search calls  {m['mean_search_calls']:.2f}")
    print(f"mean LLM calls     {m['mean_llm_calls']:.2f}")
    print(f"status             {m['status_counts']}")
    print(f"outputs            {out}/")
    bad = [r for r in recs if r["status"] != "ok"]
    if bad:
        print(f"\n{len(bad)} question(s) did not finish normally; first error ({bad[0]['status']}):")
        print(bad[0]["error"][-1500:])


if __name__ == "__main__":
    main()
