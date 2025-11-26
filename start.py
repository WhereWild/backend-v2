"""Helper script to launch the FastAPI server with reload."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    cmd = [
        "uv",
        "run",
        "uvicorn",
        "main:app",
        "--reload",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    env = dict(os.environ)
    print("Starting FastAPI via:", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


if __name__ == "__main__":
    main()
