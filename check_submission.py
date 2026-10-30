#!/usr/bin/env python3
"""Check a submission zip (or folder) with the same static checks the official harness uses.

  python check_submission.py 2023CS10000.zip
  python check_submission.py .              # check the current folder
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a4.submission_check import check_directory, check_zip  # noqa: E402


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    target = sys.argv[1]
    if target.endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmp:
            res = check_zip(target, os.path.join(tmp, "sub"))
            print(res.summary())
    else:
        res = check_directory(target)
        print(res.summary())
    sys.exit(0 if res.ok else 1)


if __name__ == "__main__":
    main()
