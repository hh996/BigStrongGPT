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


_SWANLAB_ENABLED = False


def init_swanlab(project: str, experiment_name: str, config) -> None:
    """无 SWANLAB_API_KEY 时跳过，避免本地无法开训。"""
    global _SWANLAB_ENABLED
    load_repo_dotenv()
    key = os.environ.get("SWANLAB_API_KEY")
    if not key:
        _SWANLAB_ENABLED = False
        print("[SwanLab] 未设置 SWANLAB_API_KEY，跳过实验跟踪")
        return
    # Windows 控制台 GBK 时 SwanLab 打印 emoji 会崩
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    import swanlab

    swanlab.login(api_key=key)
    cfg = vars(config) if hasattr(config, "__dict__") and not isinstance(config, dict) else config
    swanlab.init(project=project, experiment_name=experiment_name, config=cfg)
    _SWANLAB_ENABLED = True


def log_swanlab(metrics: dict) -> None:
    if not _SWANLAB_ENABLED:
        return
    import swanlab

    swanlab.log(metrics)
