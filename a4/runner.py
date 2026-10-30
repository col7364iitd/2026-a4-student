"""Runs an agent over a list of questions, exactly as the official harness does.

Design
------
* Your ``agent.py`` runs in separate *worker processes* (``spawn``), with network access disabled.
* The search engine, the LLM connection and the budget live in the trusted parent process. Your agent talks
  to them only through ``ToolsProxy`` (see ``a4/tools.py``).
* Each question has a wall-clock limit (``BUDGET.wall_clock_seconds``). If it is exceeded, the worker is
  killed and restarted, and the question scores 0 (status ``timeout``).
* An exception escaping ``run()`` scores 0 (status ``crash``). Output that is not
  ``{"answer": str, "citations": list[str]}`` scores 0 (status ``malformed``).
"""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback

from .budget import BUDGET, Budget
from .tools import Tools, ToolsProxy, dispatch

_CTX = mp.get_context("spawn")
_IMPORT_TIMEOUT = 120.0


# ====================================================================== worker (sandboxed) side
def _block_network() -> None:
    import socket

    def _blocked(*_a, **_k):
        raise PermissionError("network access is not allowed in the agent process")

    class _NoSocket(socket.socket):
        def __init__(self, *_a, **_k):
            _blocked()

    socket.socket = _NoSocket
    socket.create_connection = _blocked
    socket.getaddrinfo = _blocked


def _worker_main(conn, agent_path: str) -> None:
    _block_network()
    agent_path = os.path.abspath(agent_path)
    agent_dir = os.path.dirname(agent_path)
    if agent_dir not in sys.path:
        sys.path.append(agent_dir)  # appended, so the trusted `a4` package always wins
    try:
        spec = importlib.util.spec_from_file_location("agent", agent_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["agent"] = module
        spec.loader.exec_module(module)
        run = getattr(module, "run")
    except BaseException:
        conn.send(("import_error", traceback.format_exc(limit=8)))
        return
    conn.send(("ready", None))
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg[0] == "stop":
            return
        _, question = msg
        tools = ToolsProxy(conn)
        try:
            out = run(question, tools)
        except KeyboardInterrupt:
            return
        except BaseException:
            conn.send(("fail", traceback.format_exc(limit=8)))
            continue
        try:
            conn.send(("done", out))
        except Exception:
            conn.send(("fail", "agent output could not be serialised (return plain str / list of str)"))


# ====================================================================== trusted (parent) side
class AgentImportError(RuntimeError):
    pass


def _validate(out):
    if not isinstance(out, dict):
        return None, None, "run() must return a dict"
    ans, cits = out.get("answer"), out.get("citations", [])
    if not isinstance(ans, str):
        return None, None, "'answer' must be a string"
    if not isinstance(cits, (list, tuple)) or not all(isinstance(c, str) for c in cits):
        return None, None, "'citations' must be a list of doc_id strings"
    return ans.strip(), list(cits), None


class _Worker:
    def __init__(self, agent_path: str):
        self.agent_path = agent_path
        self.proc = None
        self.conn = None
        self.start()

    def start(self):
        parent_conn, child_conn = _CTX.Pipe(duplex=True)
        self.proc = _CTX.Process(target=_worker_main, args=(child_conn, self.agent_path), daemon=True)
        self.proc.start()
        child_conn.close()
        self.conn = parent_conn
        if not self.conn.poll(_IMPORT_TIMEOUT):
            self.kill()
            raise AgentImportError(f"agent did not finish importing within {_IMPORT_TIMEOUT:.0f}s")
        try:
            kind, payload = self.conn.recv()
        except EOFError:
            self.kill()
            raise AgentImportError("agent process exited while importing agent.py")
        if kind == "import_error":
            self.kill()
            raise AgentImportError(payload)

    def kill(self):
        try:
            if self.proc is not None and self.proc.is_alive():
                self.proc.kill()
            if self.proc is not None:
                self.proc.join(5)
        finally:
            try:
                if self.conn is not None:
                    self.conn.close()
            except Exception:
                pass

    def stop(self):
        try:
            self.conn.send(("stop",))
        except Exception:
            pass
        self.kill()


def _answer_one(worker: _Worker, q: dict, engine, llm, budget: Budget):
    tools = Tools(engine, llm, budget)
    start = time.monotonic()
    deadline = start + budget.wall_clock_seconds
    rec = {"id": q["id"], "answer": "", "citations": [], "status": "ok", "error": ""}
    restart = False
    worker.conn.send(("question", q["question"]))
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not worker.conn.poll(remaining):
            rec["status"], rec["error"] = "timeout", f"exceeded {budget.wall_clock_seconds:.0f}s wall-clock limit"
            restart = True
            break
        try:
            msg = worker.conn.recv()
        except (EOFError, OSError):
            rec["status"], rec["error"] = "crash", "agent process died (out of memory? os._exit?)"
            restart = True
            break
        kind = msg[0]
        if kind == "call":
            _, method, args, kwargs = msg
            worker.conn.send(dispatch(tools, method, args, kwargs))
        elif kind == "done":
            ans, cits, err = _validate(msg[1])
            if err:
                rec["status"], rec["error"] = "malformed", err
            else:
                rec["answer"], rec["citations"] = ans, cits
            break
        elif kind == "fail":
            rec["status"], rec["error"] = "crash", msg[1][-2000:]
            break
        else:
            rec["status"], rec["error"] = "crash", f"unexpected message {kind!r}"
            restart = True
            break
    rec["elapsed"] = round(time.monotonic() - start, 3)
    rec["infra_error"] = tools.infra_error
    rec.update(tools.usage())
    return rec, {"id": q["id"], "calls": tools.trace}, restart


def run_agent(agent_path, questions, engine, llm, workers=8, budget: Budget = BUDGET, progress=True):
    """Run ``agent_path`` on ``questions`` (dicts with 'id' and 'question').

    Returns (records, traces) in the order of ``questions``. Raises AgentImportError if agent.py cannot be
    imported at all.
    """
    if not os.path.exists(agent_path):
        raise FileNotFoundError(agent_path)
    todo: queue.Queue = queue.Queue()
    for i, q in enumerate(questions):
        todo.put((i, q))
    records = [None] * len(questions)
    traces = [None] * len(questions)
    fatal: list[BaseException] = []
    done_count = [0]
    lock = threading.Lock()
    t0 = time.monotonic()

    def loop():
        try:
            w = _Worker(agent_path)
        except BaseException as e:
            fatal.append(e)
            return
        try:
            while not fatal:
                try:
                    i, q = todo.get_nowait()
                except queue.Empty:
                    break
                rec, tr, restart = _answer_one(w, q, engine, llm, budget)
                records[i], traces[i] = rec, tr
                if restart:
                    w.kill()
                    try:
                        w = _Worker(agent_path)
                    except BaseException as e:
                        fatal.append(e)
                        return
                with lock:
                    done_count[0] += 1
                    n = done_count[0]
                if progress and (n % 10 == 0 or n == len(questions)):
                    print(f"  {n}/{len(questions)} questions  ({time.monotonic() - t0:.0f}s)", file=sys.stderr)
        finally:
            w.stop()

    threads = [threading.Thread(target=loop, daemon=True) for _ in range(max(1, min(workers, len(questions))))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if fatal and all(r is None for r in records):
        raise AgentImportError(str(fatal[0]))
    for i, q in enumerate(questions):  # questions left unanswered because a worker could not restart
        if records[i] is None:
            records[i] = {"id": q["id"], "answer": "", "citations": [], "status": "crash",
                          "error": f"not run: {fatal[0] if fatal else 'unknown'}", "elapsed": 0.0,
                          "infra_error": False, "n_search": 0, "n_llm": 0, "prompt_tokens": 0,
                          "completion_tokens": 0}
            traces[i] = {"id": q["id"], "calls": []}
    return records, traces
