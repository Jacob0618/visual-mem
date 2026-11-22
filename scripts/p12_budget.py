"""Phase 12: a hard spending cap per external baseline (default $100, user's instruction).

Every fresh (uncached) model call made for a baseline -- memory construction, query-time calls, reading, judging --
is charged to an append-only ledger results/p12_budget/<system>.jsonl. check() raises BudgetExceeded once the ledger
reaches the cap, so runners stop before spending more. Ledger appends are atomic, so several processes can share it.
"""
import json
import os
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / os.getenv("P12_LEDGER_DIR", "results/p12_budget")
CAP = float(os.getenv("P12_CAP_USD", "100"))
PRICES = {"gemini-3.8-flash": (0.75, 0.075, 3.75), "gemini-2.5-flash-lite": (0.10, 0.01, 0.40),
          "gemini-2.5-flash": (0.30, 0.03, 2.50), "gemini-embedding-001": (0.15, 0.15, 0.0)}
_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


def usd(model, prompt=0, cached=0, out=0, thoughts=0):
    pi, pc, po = PRICES.get(model, (0.75, 0.075, 3.75))
    return ((prompt - cached) * pi + cached * pc + (out + thoughts) * po) / 1e6


def charge(system, amount, what=""):
    if amount <= 0:
        return
    LEDGER.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"t": time.time(), "usd": round(amount, 8), "what": what, "pid": os.getpid()}) + "\n"
    with _lock:
        with open(LEDGER / f"{system}.jsonl", "a") as f:
            f.write(line)


def spent(system, what_prefix=""):
    p = LEDGER / f"{system}.jsonl"
    if not p.exists():
        return 0.0
    tot = 0.0
    for l in open(p):
        r = json.loads(l)
        if r["what"].startswith(what_prefix):
            tot += r["usd"]
    return tot


def check(system, cap=CAP):
    s = spent(system)
    if s >= cap:
        raise BudgetExceeded(f"{system}: ${s:.2f} spent, cap ${cap:.0f}")
    return s


def charge_gem(system, r, model, what):
    """Charge a gem.generate result unless it came from the local cache."""
    if r.get("cached"):
        return 0.0
    u = r.get("usage") or {}
    a = usd(model, u.get("prompt", 0), u.get("cached", 0), u.get("out", 0), u.get("thoughts", 0))
    charge(system, a, what)
    return a
