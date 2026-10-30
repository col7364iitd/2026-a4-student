#!/usr/bin/env python3
"""Build <entry_number>.zip for Moodle: agent.py, helpers/ (if present) and report.pdf, then check it.

  python make_submission.py 2023CS10000
  python make_submission.py 2023CS10000 --no-report      # leaderboard submissions: code only
"""

import argparse
import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a4.submission_check import IGNORED, check_zip  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entry", help="your entry number, e.g. 2023CS10000")
    ap.add_argument("--no-report", action="store_true", help="leave out report.pdf (leaderboard submissions)")
    args = ap.parse_args()

    files = ["agent.py"]
    if os.path.isdir("helpers"):
        for dp, dns, fns in os.walk("helpers"):
            dns[:] = [d for d in dns if d not in IGNORED]
            files += [os.path.join(dp, f) for f in fns if f.endswith(".py")]
    if not args.no_report:
        if os.path.exists("report.pdf"):
            files.append("report.pdf")
        else:
            print("WARNING: report.pdf not found - the final Moodle submission must include it.")

    out = f"{args.entry}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f)
    print(f"Wrote {out} with {len(files)} file(s): {', '.join(files)}\n")
    with tempfile.TemporaryDirectory() as tmp:
        res = check_zip(out, os.path.join(tmp, "sub"))
    print(res.summary())
    sys.exit(0 if res.ok else 1)


if __name__ == "__main__":
    main()
