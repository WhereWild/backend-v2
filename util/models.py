from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from util.config import load_config


CONFIG = load_config("global")
_MODEL_KIND = str(CONFIG.ml_model_kind).strip().lower() or "gbt"
AUTO_MODEL_ID = f"auto_{_MODEL_KIND}_sdm"
AUTO_PHENOLOGY_MODEL_ID = f"auto_{_MODEL_KIND}_phenology"
AUTO_FULL_MODEL_ID = f"auto_{_MODEL_KIND}_full"
DEFAULT_MODEL_ID = AUTO_MODEL_ID


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    model_dir: Path
    payload: dict[str, Any]


def _models_root() -> Path:
    override = os.environ.get("WHEREWILD_MODEL_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return CONFIG.models_root.resolve()


def _iter_model_artifact_dirs() -> list[Path]:
    root = _models_root()
    if not root.exists():
        return []
    dirs: list[Path] = []
    for child in root.iterdir():
        if child.is_dir() and (child / "model.pkl").exists():
            dirs.append(child)
    dirs.sort(key=lambda path: path.name, reverse=True)
    return dirs


def _latest_artifact_for_prefix(prefix: str) -> Path | None:
    dirs = _iter_model_artifact_dirs()
    # Exact match wins over prefix match
    for model_dir in dirs:
        if model_dir.name == prefix:
            return model_dir
    for model_dir in dirs:
        if model_dir.name.startswith(prefix + "_"):
            return model_dir
    return None


def _resolve_model_dir(model_id: str | None, taxon_id: str | int | None) -> Path | None:
    normalized = (model_id or "").strip()
    taxon_key = str(taxon_id).strip() if taxon_id is not None else ""

    if normalized == "":
        if not taxon_key:
            return None
        model_kind = str(CONFIG.ml_model_kind).strip().lower() or "gbt"
        return _latest_artifact_for_prefix(f"taxon_{taxon_key}_{model_kind}")

    if normalized.startswith("auto_"):
        # Suffix after "auto_" is the full kind+mode string, e.g. "gbt_sdm" or "gbt_phenology".
        kind_and_mode = normalized.removeprefix("auto_").strip().lower()
        if not kind_and_mode or not taxon_key:
            return None
        # New-style: taxon_{key}_gbt_sdm_TIMESTAMP or taxon_{key}_gbt_sdm
        result = _latest_artifact_for_prefix(f"taxon_{taxon_key}_{kind_and_mode}")
        if result is not None:
            return result
        # Old-style fallback ONLY for sdm: taxon_{key}_gbt (no mode suffix, pre-refactor artifacts)
        if "_" in kind_and_mode and kind_and_mode.rsplit("_", 1)[1] == "sdm":
            base_kind = kind_and_mode.rsplit("_", 1)[0]
            return _latest_artifact_for_prefix(f"taxon_{taxon_key}_{base_kind}")
        return None

    if normalized.startswith("taxon_"):
        # Exact match first (full artifact id passed), then prefix search
        exact = _models_root() / normalized
        if exact.is_dir() and (exact / "model.pkl").exists():
            return exact
        return _latest_artifact_for_prefix(f"{normalized}_")

    candidate = _models_root() / normalized
    if candidate.is_dir() and (candidate / "model.pkl").exists():
        return candidate
    return None


@lru_cache(maxsize=32)
def _load_model_payload(model_dir: str) -> dict[str, Any]:
    model_path = Path(model_dir) / "model.pkl"
    with open(model_path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid model payload type at {model_path}")
    if "model" not in payload or "preprocessor" not in payload:
        raise ValueError(f"Model payload missing required keys at {model_path}")
    return payload


@lru_cache(maxsize=64)
def _load_json_file(path: str) -> dict[str, Any] | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def resolve_model_artifact(
    model_id: str | None,
    *,
    taxon_id: str | int | None,
) -> ModelArtifact | None:
    model_dir = _resolve_model_dir(model_id, taxon_id)
    if model_dir is None:
        return None
    return ModelArtifact(
        model_id=model_dir.name,
        model_dir=model_dir,
        payload=_load_model_payload(str(model_dir)),
    )


def model_feature_columns(
    model_id: str | None,
    *,
    taxon_id: str | int | None,
) -> list[str]:
    artifact = resolve_model_artifact(model_id, taxon_id=taxon_id)
    if artifact is None:
        return []
    return [str(col) for col in artifact.payload.get("feature_columns") or [] if str(col).strip()]



def describe_model(
    model_id: str | None,
    *,
    taxon_id: str | int | None,
) -> dict[str, Any]:
    artifact = resolve_model_artifact(model_id, taxon_id=taxon_id)
    if artifact is None:
        return {
            "available": False,
            "requested_model_id": (model_id or "").strip() or AUTO_MODEL_ID,
        }

    summary = _load_json_file(str(artifact.model_dir / "summary.json")) or {}
    metrics = _load_json_file(str(artifact.model_dir / "metrics.json")) or {}
    return {
        "available": True,
        "requested_model_id": (model_id or "").strip() or AUTO_MODEL_ID,
        "resolved_model_id": artifact.model_id,
        "model_dir": str(artifact.model_dir),
        "taxon_id": str(taxon_id) if taxon_id is not None else None,
        "feature_columns": list(artifact.payload.get("feature_columns") or []),
        "summary": summary,
        "metrics": metrics,
        "phenology_available": has_phenology_model(taxon_id),
        "full_available": has_full_model(taxon_id),
    }


def has_sdm_model(taxon_id: str | int | None) -> bool:
    return resolve_model_artifact(AUTO_MODEL_ID, taxon_id=taxon_id) is not None


def has_phenology_model(taxon_id: str | int | None) -> bool:
    return resolve_model_artifact(AUTO_PHENOLOGY_MODEL_ID, taxon_id=taxon_id) is not None


def has_full_model(taxon_id: str | int | None) -> bool:
    return resolve_model_artifact(AUTO_FULL_MODEL_ID, taxon_id=taxon_id) is not None


def get_all_sdm_taxon_ids() -> list[int]:
    """Return taxon IDs for all taxa that have a trained SDM artifact."""
    import re
    model_kind = str(CONFIG.ml_model_kind).strip().lower() or "gbt"
    # Match taxon_{id}_{kind}_sdm* or old-style taxon_{id}_{kind} (no mode suffix)
    pattern = re.compile(
        rf"^taxon_(\d+)_{re.escape(model_kind)}(?:_sdm|$)"
    )
    seen: dict[int, None] = {}
    for artifact_dir in _iter_model_artifact_dirs():
        m = pattern.match(artifact_dir.name)
        if m:
            seen[int(m.group(1))] = None
    return list(seen.keys())


def predict(
    model_id: str | None,
    features: np.ndarray,
    *,
    feature_ids: Sequence[str] | None = None,
    taxon_id: str | int | None = None,
    _preflat: "np.ndarray | None" = None,
    _valid_mask: "np.ndarray | None" = None,
) -> np.ndarray:
    artifact = resolve_model_artifact(model_id, taxon_id=taxon_id)
    if artifact is None:
        requested = (model_id or "").strip() or AUTO_MODEL_ID
        raise ValueError(f"No model artifact found for model_id='{requested}' taxon_id='{taxon_id}'")
    return _predict_sklearn_artifact(
        artifact.payload,
        features,
        feature_ids=feature_ids,
        _preflat=_preflat,
        _valid_mask=_valid_mask,
    )


def _predict_sklearn_artifact(
    payload: dict[str, Any],
    features: np.ndarray,
    *,
    feature_ids: Sequence[str] | None,
    _preflat: "np.ndarray | None" = None,
    _valid_mask: "np.ndarray | None" = None,
) -> np.ndarray:
    if features.ndim != 3:
        raise ValueError("Expected feature tensor with shape (H, W, C)")
    if features.size == 0:
        return np.zeros(features.shape[:-1], dtype=np.float32)

    h, w, c = features.shape
    expected_columns = [str(col) for col in payload.get("feature_columns") or [] if str(col).strip()]
    if not expected_columns:
        raise ValueError("Model payload missing feature_columns")

    channel_ids = list(feature_ids) if feature_ids else []
    if channel_ids and len(channel_ids) != c:
        raise ValueError(
            f"feature_ids length ({len(channel_ids)}) does not match channel count ({c})"
        )
    if not channel_ids:
        raise ValueError("feature_ids are required for model prediction")

    flat = _preflat if _preflat is not None else features.reshape(-1, c).astype(np.float32, copy=False)

    # Vectorized column selection — one fancy-index op instead of a Python loop
    channel_idx = {name: idx for idx, name in enumerate(channel_ids)}
    src_indices = np.array([channel_idx.get(name, -1) for name in expected_columns], dtype=np.intp)
    matrix = np.full((flat.shape[0], len(expected_columns)), np.nan, dtype=np.float32)
    valid_cols = src_indices >= 0
    if valid_cols.any():
        matrix[:, valid_cols] = flat[:, src_indices[valid_cols]]

    valid_mask = _valid_mask if _valid_mask is not None else np.any(np.isfinite(flat), axis=1)
    frame = pd.DataFrame(matrix, columns=expected_columns)
    transformed = payload["preprocessor"].transform(frame)
    probs = payload["model"].predict_proba(transformed)[:, 1].astype(np.float32)
    probs[~valid_mask] = np.nan
    return probs.reshape(h, w)
