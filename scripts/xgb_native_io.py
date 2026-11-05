"""Load the verified native XGBoost artifact without unpickling XGBoost objects."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NativeBinaryXGB:
    """Expose predict/predict_proba for a native binary-logistic Booster."""

    def __init__(self, booster):
        self.booster = booster
        config = json.loads(booster.save_config())
        objective = config["learner"]["objective"]["name"]
        if objective != "binary:logistic":
            raise ValueError(f"Unsupported XGBoost objective: {objective}")
        self.n_features_in_ = int(booster.num_features())

    def predict_proba(self, features):
        import xgboost as xgb

        matrix = xgb.DMatrix(
            np.asarray(features, dtype=np.float32),
            feature_names=self.booster.feature_names,
        )
        positive = np.asarray(self.booster.predict(matrix), dtype=np.float64)
        if positive.shape != (len(features),):
            raise ValueError(f"Unexpected XGBoost probability shape: {positive.shape}")
        return np.column_stack((1.0 - positive, positive))

    def predict(self, features):
        return (self.predict_proba(features)[:, 1] > 0.5).astype(np.int64)


def load_verified_xgb(models_dir: Path):
    """Use JSON only after a matching source/hash and prediction check are recorded.

    Falls back to the original bundle if conversion has not been performed.
    A broken or tampered conversion is a hard error, not a silent fallback.
    """
    import joblib

    directory = Path(models_dir) / "xgboost"
    legacy = directory / "xgb_model.joblib"
    native = directory / "xgb_model.json"
    manifest_path = directory / "xgb_conversion_verified.json"

    if not manifest_path.exists() and not native.exists():
        return joblib.load(legacy)
    if not manifest_path.is_file() or not native.is_file():
        raise RuntimeError("Incomplete XGBoost JSON conversion; retain/use the original files until verified")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "verified" or manifest.get("schema_version") != 1:
        raise ValueError("XGBoost conversion has not passed verification")
    if manifest.get("source_sha256") != sha256_file(legacy):
        raise ValueError("Original XGBoost model differs from the verified source")
    if manifest.get("native_sha256") != sha256_file(native):
        raise ValueError("XGBoost JSON checksum mismatch")
    metadata = manifest.get("metadata", {})
    if (metadata.get("feature_policy") != "no_duration"
            or len(metadata.get("feature_names", [])) != 152
            or metadata.get("label_map") != {"normal": 0, "abnormal": 1}):
        raise ValueError("Invalid XGBoost metadata in conversion manifest")

    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(native))
    model = NativeBinaryXGB(booster)
    if model.n_features_in_ != len(metadata["feature_names"]):
        raise ValueError("Native XGBoost model feature count differs from manifest")
    return {"model": model, **metadata}
