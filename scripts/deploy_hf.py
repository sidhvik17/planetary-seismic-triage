"""Deploy the Streamlit demo to a Hugging Face Space (free CPU tier).

Usage:
  set HF_TOKEN env var (WRITE token from hf.co/settings/tokens), then
  python scripts/deploy_hf.py --repo <username>/planetary-seismic-triage

Uploads only what the app needs: app/, planetseis/, models/, demo_data/,
.streamlit/, a slim requirements.txt, and a Space README with the required
YAML front-matter. The 2 GB data packet, caches, and training runs stay home.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, upload_folder

ROOT = Path(__file__).resolve().parent.parent

SPACE_README = """---
title: Planetary Seismic Triage
emoji: 🌒
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 8501
pinned: false
---

# Planetary Seismic Event Detection

Lightweight (118K-param) dual-head 1D CNN detecting seismic events in Apollo
(Moon) and InSight (Mars) recordings, with MC-Dropout uncertainty and an
on-lander downlink-triage simulation. B.Tech major project; $0 stack.

Upload a miniSEED/SAC/CSV trace in the **Analyze** tab, or press
**Start stream** in the **Triage** tab for the mission simulation on a real
held-out Apollo day.
"""

# CPU torch keeps the Space build small and fast.
SPACE_REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.6
numpy>=1.26
scipy>=1.11
obspy>=1.4
pandas>=2.0
scikit-learn>=1.4
streamlit>=1.37
plotly>=5.20
"""

# Non-root runtime user (uid 1000, as Hugging Face Spaces expects).
DOCKERFILE = """FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
USER user
ENV HOME=/home/user STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", \\
            "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
"""

INCLUDE = ["app", "planetseis", "models", "demo_data", ".streamlit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. username/planetary-seismic-triage")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN env var not set — create a WRITE token at "
                 "https://huggingface.co/settings/tokens")

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="space", space_sdk="docker", exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        stage = Path(td)
        for name in INCLUDE:
            src = ROOT / name
            if not src.exists():
                print(f"skip missing {name}")
                continue
            dst = stage / name
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.rglob("*"):
                if f.is_file() and "__pycache__" not in f.parts:
                    rel = f.relative_to(src)
                    (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                    (dst / rel).write_bytes(f.read_bytes())
        (stage / "README.md").write_text(SPACE_README, encoding="utf-8")
        (stage / "requirements.txt").write_text(SPACE_REQUIREMENTS, encoding="utf-8")
        (stage / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
        print("uploading to Space:", args.repo)
        upload_folder(repo_id=args.repo, repo_type="space", folder_path=str(stage),
                      token=token, commit_message="Deploy planetary seismic triage demo")
    print(f"done -> https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
