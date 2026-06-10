"""
Container launcher: pull the latest app code from a Git repo onto the
persistent volume, then run the dashboard.

This decouples CODE from the Docker IMAGE. To ship a code change you just
`git push` and click "Restart" on the pod — no rebuild, no re-push, no pod
recreation, and (with the volume) no model re-download.

Behaviour:
  - If REPO_URL is set: clone it to /workspace/app on first boot, otherwise
    hard-reset it to the latest remote commit on every (re)start.
  - If REPO_URL is not set: fall back to the code baked into /app.
"""

import os
import subprocess
import sys

REPO_URL = os.environ.get("REPO_URL")
BRANCH = os.environ.get("REPO_BRANCH", "main")
APP_DIR = "/workspace/app"


def run(cmd, **kw):
    print("[start] $ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


if REPO_URL:
    if not os.path.isdir(os.path.join(APP_DIR, ".git")):
        print(f"[start] cloning {REPO_URL} ({BRANCH}) -> {APP_DIR}", flush=True)
        run(["git", "clone", "--branch", BRANCH, REPO_URL, APP_DIR], check=True)
    else:
        print("[start] syncing to latest remote code...", flush=True)
        run(["git", "-C", APP_DIR, "fetch", "origin", BRANCH], check=False)
        run(["git", "-C", APP_DIR, "reset", "--hard", f"origin/{BRANCH}"], check=False)
    workdir = APP_DIR
else:
    print("[start] REPO_URL not set; using baked code in /app", flush=True)
    workdir = "/app"

os.chdir(workdir)
print(f"[start] launching dashboard from {workdir}", flush=True)
os.execvp(sys.executable, [sys.executable, "dashboard/app.py"])
