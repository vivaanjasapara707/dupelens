"""Deploy the DupeLens website to Vercel, with every /api/... call forwarded to the live app on Hugging Face.

Vercel serves the pages in dashboard/ (fast, free); the models (3.3 GB) cannot run on Vercel, so its
rewrites send /api/* and /uploads/* to the Hugging Face Space that runs src/app.py (scripts/deploy_hf.py).
Without a Space URL the site still works with the saved example searches and the insights page.

Needs VERCEL_TOKEN (vercel.com/account/tokens) in .env or the environment.
Run from the repo root:  .venv\\Scripts\\python.exe scripts\\deploy_vercel.py https://<user>-dupelens.hf.space
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402


def main():
    token = config.env("VERCEL_TOKEN")
    if not token:
        raise SystemExit("Put VERCEL_TOKEN=... (vercel.com/account/tokens) in .env")
    backend = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else None
    site = ROOT / "dashboard"
    cfg = {"cleanUrls": False}
    if backend:
        cfg["rewrites"] = [{"source": "/api/:path*", "destination": f"{backend}/api/:path*"},
                           {"source": "/uploads/:path*", "destination": f"{backend}/uploads/:path*"}]
    (site / "vercel.json").write_text(json.dumps(cfg, indent=2))
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    cmd = [npx, "vercel", "deploy", "--prod", "--yes", "--token", token, "--name", "dupelens"]
    print("deploying", site, "->", "live API at " + backend if backend else "static (saved examples only)")
    subprocess.run(cmd, cwd=site, check=True)


if __name__ == "__main__":
    main()
