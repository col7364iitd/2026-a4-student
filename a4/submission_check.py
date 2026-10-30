"""Static checks on a submission (handout Sections 7 and 8). Used by ``check_submission.py`` and by the
official harness, so a submission that passes here locally passes the same checks when graded.

Errors make the submission invalid. Warnings are reviewed by staff (e.g. large literals that could hold
stored answers) and may lead to a viva.
"""

from __future__ import annotations

import ast
import os
import shutil
import zipfile
from dataclasses import dataclass, field

MAX_CODE_BYTES = 100 * 1024

ALLOWED_MODULES = frozenset(
    """a4 numpy re json math string collections itertools functools operator dataclasses typing heapq
    bisect unicodedata difflib statistics textwrap enum abc copy time random __future__""".split()
)
FORBIDDEN_CALLS = frozenset({"eval", "exec", "compile", "__import__", "open", "breakpoint", "input"})
FORBIDDEN_ATTRS = frozenset(
    {"__subclasses__", "__globals__", "__builtins__", "__code__", "__closure__", "__dict__", "__mro__",
     "__bases__", "__loader__", "__spec__", "__getattribute__", "f_globals", "f_locals", "f_back", "gi_frame", "_conn"}
)
IGNORED = ("__pycache__", "__MACOSX", ".DS_Store", ".git", ".ipynb_checkpoints")
LARGE_STRING = 1500       # characters in one literal
LARGE_COLLECTION = 100    # elements in one dict/list/set/tuple literal
TOTAL_STRING = 20000      # characters over all literals


@dataclass
class CheckResult:
    root: str = ""
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    code_bytes: int = 0
    has_report: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [f"Submission root: {self.root}", f"Python code size: {self.code_bytes} bytes (limit {MAX_CODE_BYTES})",
                 f"report.pdf present: {self.has_report}"]
        lines += [f"ERROR:   {e}" for e in self.errors]
        lines += [f"WARNING: {w}" for w in self.warnings]
        lines.append("RESULT:  " + ("OK" if self.ok else "INVALID - fix the errors above"))
        return "\n".join(lines)


def _ignored(rel: str) -> bool:
    return any(part in IGNORED or part.startswith("._") for part in rel.split(os.sep))


def _local_modules(root: str) -> set[str]:
    mods = set()
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if name.endswith(".py"):
            mods.add(name[:-3])
        elif os.path.isdir(p) and not _ignored(name):
            mods.add(name)
    return mods


def _check_ast(path: str, rel: str, local: set[str], res: CheckResult, totals: dict) -> None:
    try:
        with open(path, encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src, filename=rel)
    except (SyntaxError, UnicodeDecodeError) as e:
        res.errors.append(f"{rel}: cannot be parsed ({e})")
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top not in ALLOWED_MODULES and top not in local:
                    res.errors.append(f"{rel}:{node.lineno}: import of '{a.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                top = (node.module or "").split(".")[0]
                if top not in ALLOWED_MODULES and top not in local:
                    res.errors.append(f"{rel}:{node.lineno}: import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            res.errors.append(f"{rel}:{node.lineno}: call to '{node.func.id}()' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            res.errors.append(f"{rel}:{node.lineno}: access to '.{node.attr}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_ATTRS:
            res.errors.append(f"{rel}:{node.lineno}: use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            if isinstance(node.value, str) and node.value in FORBIDDEN_ATTRS | {"__getattribute__"}:
                res.errors.append(f"{rel}:{node.lineno}: reference to '{node.value}' is not allowed")
            n = len(node.value)
            totals["chars"] += n
            if n > LARGE_STRING:
                res.warnings.append(f"{rel}:{node.lineno}: large string literal ({n} chars) - will be reviewed")
        elif isinstance(node, (ast.Dict, ast.List, ast.Set, ast.Tuple)):
            n = len(node.keys) if isinstance(node, ast.Dict) else len(node.elts)
            if n > LARGE_COLLECTION:
                res.warnings.append(f"{rel}:{node.lineno}: literal with {n} elements - will be reviewed")
    if rel == "agent.py":
        runs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"]
        if not runs:
            res.errors.append("agent.py: no top-level function 'run(question, tools)'")
        elif len(runs[0].args.args) < 2:
            res.errors.append("agent.py: run() must accept (question, tools)")


def check_directory(root: str) -> CheckResult:
    res = CheckResult(root=os.path.abspath(root))
    if not os.path.isfile(os.path.join(root, "agent.py")):
        res.errors.append("agent.py not found at the top level of the submission")
        return res
    if os.path.isdir(os.path.join(root, "a4")) or os.path.isfile(os.path.join(root, "a4.py")):
        res.errors.append("the submission must not contain an 'a4' package/module (the harness provides it)")
    local = _local_modules(root)
    totals = {"chars": 0}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _ignored(d)]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            if _ignored(rel):
                continue
            if fn.endswith(".py"):
                res.code_bytes += os.path.getsize(full)
                _check_ast(full, rel, local, res, totals)
            elif rel == "report.pdf":
                res.has_report = True
            else:
                res.errors.append(f"file not allowed in submission: {rel} (only .py files and report.pdf)")
    if res.code_bytes > MAX_CODE_BYTES:
        res.errors.append(f"Python code is {res.code_bytes} bytes; the limit is {MAX_CODE_BYTES}")
    if totals["chars"] > TOTAL_STRING:
        res.warnings.append(f"string literals total {totals['chars']} chars - will be reviewed")
    return res


def extract_zip(zip_path: str, dest: str) -> str:
    """Extract safely and return the submission root (the folder that contains agent.py)."""
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    with zipfile.ZipFile(zip_path) as z:
        total = 0
        for info in z.infolist():
            name = info.filename
            if name.startswith("/") or ".." in name.split("/"):
                raise ValueError(f"unsafe path in zip: {name}")
            total += info.file_size
            if total > 50 * 1024 * 1024:
                raise ValueError("zip expands to more than 50 MB")
        z.extractall(dest)
    if os.path.isfile(os.path.join(dest, "agent.py")):
        return dest
    subdirs = [d for d in os.listdir(dest) if os.path.isdir(os.path.join(dest, d)) and not _ignored(d)]
    if len(subdirs) == 1 and os.path.isfile(os.path.join(dest, subdirs[0], "agent.py")):
        return os.path.join(dest, subdirs[0])
    return dest


def check_zip(zip_path: str, dest: str) -> CheckResult:
    try:
        root = extract_zip(zip_path, dest)
    except (zipfile.BadZipFile, ValueError) as e:
        r = CheckResult(root=zip_path)
        r.errors.append(f"cannot open zip: {e}")
        return r
    res = check_directory(root)
    if root != dest:
        res.warnings.append("files were inside a sub-folder of the zip; put agent.py at the top level next time")
    return res
