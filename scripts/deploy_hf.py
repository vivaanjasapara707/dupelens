"""Deploy DupeLens to a Hugging Face Space (Docker) in one command.

Needs a Hugging Face token with WRITE access in .env (HF_TOKEN=...) or the environment, plus
SERPER_API_KEY (and optionally GROQ_API_KEY) in .env; those are stored as private Space secrets,
never uploaded as files.
Run from the repo root:  .venv\\Scripts\\python.exe scripts\\deploy_hf.py [space-name]
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

SPACE_README = """---
title: DupeLens
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Same product, many sellers - find where to buy it
---
DupeLens: search any product sold online in India; a trained matcher keeps only listings of the same item
and shows the fair price, the sort order of sellers and where to buy. Code and evaluation:
https://github.com/vivaanjasapara707/dupelens
"""


def main():
    token = config.env("HF_TOKEN")
    if not token:
        raise SystemExit("Put HF_TOKEN=... (a WRITE token from huggingface.co/settings/tokens) in .env")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo = f"{user}/{sys.argv[1] if len(sys.argv) > 1 else 'dupelens'}"
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
    for name in ("SERPER_API_KEY", "GROQ_API_KEY"):
        if config.env(name):
            api.add_space_secret(repo, name, config.env(name))
    api.upload_folder(repo_id=repo, repo_type="space", folder_path=str(ROOT),
                      ignore_patterns=[".git/*", ".venv/*", ".env*", ".claude/*", "data/*", "cache/live/*", "cache/insights/*",
                                       "cache/uploads/*", "results/labelling/*", "notebooks/*", "**/__pycache__/*", "README.md"])
    api.upload_file(path_or_fileobj=SPACE_README.encode(), path_in_repo="README.md", repo_id=repo, repo_type="space")
    app_url = f"https://{repo.replace('/', '-').replace('_', '-').lower()}.hf.space"
    print(f"Deployed: https://huggingface.co/spaces/{repo}  (the app: {app_url})")
    print(f"Website on Vercel with the live API:  python scripts/deploy_vercel.py {app_url}")


if __name__ == "__main__":
    main()
