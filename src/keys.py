import os
from pathlib import Path

_ENV_CANDIDATES = [Path(__file__).resolve().parents[1] / ".env", Path.home() / "Desktop/jevmem/.env"]


def gemini_key() -> str:
    k = os.environ.get("GEMINI_API_KEY")
    if k:
        return k
    for p in _ENV_CANDIDATES:
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip().startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("GEMINI_API_KEY not found")


def openrouter_key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    for p in _ENV_CANDIDATES:
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip().startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("OPENROUTER_API_KEY not found")
