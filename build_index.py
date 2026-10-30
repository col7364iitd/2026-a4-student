#!/usr/bin/env python3
"""(Re)build the BM25 index from data/corpus.jsonl.

You normally do NOT need this: the data bundle already contains data/bm25/. Use it only if the index is
missing or you were told to rebuild it. The parameters are fixed (k1=0.9, b=0.4); the official harness uses
the same index.

  python build_index.py --data-dir data
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a4.bm25 import BM25Index  # noqa: E402
from a4.data import Corpus  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    t0 = time.time()
    corpus = Corpus(os.path.join(args.data_dir, "corpus.jsonl"))
    print(f"Indexing {len(corpus)} paragraphs ...")
    index = BM25Index.build(corpus.index_text(i) for i in range(len(corpus)))
    index.save(os.path.join(args.data_dir, "bm25"))
    print(f"Done in {time.time() - t0:.1f}s: {len(index.vocab)} terms -> {args.data_dir}/bm25/")


if __name__ == "__main__":
    main()
