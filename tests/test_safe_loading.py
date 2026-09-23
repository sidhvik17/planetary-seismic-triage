"""Checkpoint loading must not unpickle arbitrary objects.

Model-weight files load with weights_only=True everywhere. The only allowed
exception is the trainer's own resume file (last.pt), whose NumPy RNG state
needs full unpickling.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {Path("planetseis/training_state.py")}


def test_only_the_resume_loader_unpickles_objects():
    offenders = []
    for folder in ("app", "planetseis", "scripts"):
        for path in (ROOT / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            relative = path.relative_to(ROOT)
            if "weights_only=False" in text and relative not in ALLOWED:
                offenders.append(str(relative))
            # every torch.load call states its choice explicitly
            for call in re.finditer(r"torch\.load\(([^()]|\([^()]*\))*\)", text, re.S):
                if "weights_only" not in call.group(0):
                    offenders.append(f"{relative}: torch.load without weights_only")
    assert offenders == []
