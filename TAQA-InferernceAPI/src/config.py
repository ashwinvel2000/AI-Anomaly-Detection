from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

try:
    # Load .env if present
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def _strip_inline_comment(s: str) -> str:
    # Remove inline comments starting with space-hash or tab-hash
    for sep in [" #", "\t#"]:
        idx = s.find(sep)
        if idx != -1:
            return s[:idx]
    return s


def _get_path(key: str, default: str) -> Path:
    raw = os.getenv(key, default)
    # Normalize: drop inline comments, strip quotes/spaces and expand user/env vars
    raw = _strip_inline_comment(raw).strip().strip('"').strip("'")
    return Path(os.path.expandvars(os.path.expanduser(raw)))


# External, read-only inference models (e.g., wwwroot/models)
EXTERNAL_MODEL_DIR: Path = _get_path(
    "EXTERNAL_MODEL_DIR",
    str(Path("wwwroot") / "models"),
)

# Internal training registry for new versions
REGISTRY_DIR: Path = _get_path(
    "REGISTRY_DIR",
    str(Path("models")),
)

# Which source to use for inference by default (reserved for future use)
DEFAULT_INFERENCE_SOURCE: Literal["external", "registry"] = (
    os.getenv("DEFAULT_INFERENCE_SOURCE", "external").strip().lower()  # type: ignore
    if os.getenv("DEFAULT_INFERENCE_SOURCE") else "external"
)

# Promotion gate toggle
ALLOW_PROMOTE: bool = os.getenv("ALLOW_PROMOTE", "false").strip().lower() in {"1", "true", "yes", "on"}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# Ensure registry directory exists (safe no-op if already there)
try:
    ensure_dir(REGISTRY_DIR)
except Exception:
    pass
