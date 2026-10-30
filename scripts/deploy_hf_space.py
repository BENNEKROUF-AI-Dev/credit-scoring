"""Déploie l'API sur un Hugging Face Space (SDK Docker). Appelé par la CI/CD.

Variables d'environnement : HF_TOKEN (secret GitHub), HF_SPACE (ex: pseudo/credit-scoring-api)
"""
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
SPACE_README = """---
title: Credit Scoring API
emoji: 💳
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
API de scoring « Prêt à Dépenser » — Swagger : /docs — Interface : /ui
"""


def main():
    token, space = os.environ["HF_TOKEN"], os.environ["HF_SPACE"]
    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for item in ("app", "model", "Dockerfile", "requirements.txt", ".dockerignore"):
            src = ROOT / item
            (shutil.copytree if src.is_dir() else shutil.copy)(src, tmp / item)
        (tmp / "README.md").write_text(SPACE_README)
        api.upload_folder(repo_id=space, repo_type="space", folder_path=str(tmp),
                          commit_message=f"Deploy {os.getenv('GITHUB_SHA', 'local')[:7]}",
                          ignore_patterns=["**/__pycache__/**"])
    print(f"Déployé : https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main()
