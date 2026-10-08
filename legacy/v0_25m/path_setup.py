"""Legacy v0.25M：把仓库根目录与 legacy 包加入 sys.path，并解析共享 dataset 路径。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

LEGACY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = LEGACY_ROOT.parent.parent


def setup_import_paths() -> None:
    for root in (LEGACY_ROOT, REPO_ROOT):
        s = str(root)
        if s not in sys.path:
            sys.path.insert(0, s)


def repo_dataset(*parts: str) -> str:
    return str(REPO_ROOT.joinpath("dataset", *parts))


def legacy_output(*parts: str) -> str:
    return str(LEGACY_ROOT.joinpath("output", *parts))


def load_repo_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_file = REPO_ROOT / ".env"
    if env_file.is_file():
        load_dotenv(env_file)
