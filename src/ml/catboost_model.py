"""
Tree-based models for pitch prediction.

This module provides a CatBoost baseline and an XGBoost-compatible variant
that share the same feature preparation and evaluation pipeline.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    top_k_accuracy_score,
)


class PitchCatBoostModel:
    """
    CatBoost-based pitch prediction model.

    Uses separate models for:
    - Pitch type classification (CatBoostClassifier)
    - Location prediction (CatBoostRegressor for px and pz)
    """

    def __init__(
        self,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        depth: int = 8,
        l2_leaf_reg: float = 3.0,
        early_stopping_rounds: int = 50,
        task_type: str = "CPU",
        random_seed: int = 42,
        verbose: int = 100,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.early_stopping_rounds = early_stopping_rounds
        self.task_type = task_type
        self.random_seed = random_seed
        self.verbose = verbose

        self.type_model: Optional[CatBoostClassifier] = None
        self.px_model: Optional[CatBoostRegressor] = None
        self.pz_model: Optional[CatBoostRegressor] = None

        self.feature_columns: list[str] = []
        self.categorical_features: list[str] = []

    def _create_type_model(self, n_classes: int) -> CatBoostClassifier:
        return CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function="MultiClass",
            classes_count=n_classes,
            task_type=self.task_type,
            random_seed=self.random_seed,
            verbose=self.verbose,
            early_stopping_rounds=self.early_stopping_rounds,
        )

    def _create_location_model(self) -> CatBoostRegressor:
        return CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function="RMSE",
            task_type=self.task_type,
            random_seed=self.random_seed,
            verbose=self.verbose,
            early_stopping_rounds=self.early_stopping_rounds,
        )

    def get_feature_columns(self) -> list[str]:
        return [
            "balls",
            "strikes",
            "two_strike_count",
            "hitters_count",
            "first_pitch",
            "pitcher_ahead",
            "inning",
            "outs",
            "runners_bitmap",
            "score_diff",
            "pitcher_id",
            "batter_id",
            "throw_side",
            "bat_side",
            "platoon_same_side",
            "pitcher_ff_pct",
            "pitcher_repertoire",
            "batter_zone_height",
            "batter_zone_mid",
            "prev_pitch_type",
            "prev_px",
            "prev_pz",
            "prev_speed",
            "prev_is_strike",
            "velocity_delta",
            "prev_swing",
            "prev_result_type",
            "n_fastballs_in_ab",
            "n_breaking_in_ab",
            "same_pitch_streak",
            "pitch_number",
            "temp_normalized",
            "wind_speed",
            "wind_direction",
            "is_night_game",
            "season_progress",
            "runners_in_scoring_position",
            "leverage_approx",
            "pitcher_pitch_count",
        ]

    def get_categorical_features(self) -> list[str]:
        return [
            "pitcher_id",
            "batter_id",
            "throw_side",
            "bat_side",
            "prev_pitch_type",
        ]

    def prepare_data(
        self,
        df: pl.DataFrame,
        feature_engine,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
        from src.ml.features import PITCH_TYPE_TO_IDX

        df = feature_engine.transform(df)
        df = df.with_columns([
            pl.col("pitch_type_code")
            .shift(1)
            .over(["game_pk", "at_bat_index"])
            .fill_null("NONE")
            .alias("prev_pitch_type"),
        ])

        df = df.filter(
            pl.col("pitch_type_idx").is_not_null()
            & pl.col("px").is_not_null()
            & pl.col("pz").is_not_null()
        )

        feature_cols = self.get_feature_columns()
        cat_cols = self.get_categorical_features()

        available_cols = []
        for col in feature_cols:
            if col in df.columns:
                available_cols.append(col)
            else:
                print(f"Warning: Column {col} not found, skipping")

        self.feature_columns = available_cols
        cat_indices = [i for i, col in enumerate(available_cols) if col in cat_cols]
        self.categorical_features = [available_cols[i] for i in cat_indices]

        X = df.select(available_cols).to_pandas()
        y_type = df["pitch_type_idx"].to_numpy()
        y_px = df["px"].to_numpy()
        y_pz = df["pz"].to_numpy()

        return X, y_type, y_px, y_pz, cat_indices

    def train(
        self,
        X_train,
        y_type_train: np.ndarray,
        y_px_train: np.ndarray,
        y_pz_train: np.ndarray,
        X_val,
        y_type_val: np.ndarray,
        y_px_val: np.ndarray,
        y_pz_val: np.ndarray,
        cat_features: list[int],
        n_classes: int = 11,
    ) -> dict:
        results = {}

        train_pool_type = Pool(X_train, y_type_train, cat_features=cat_features)
        val_pool_type = Pool(X_val, y_type_val, cat_features=cat_features)

        print("\n" + "=" * 60)
        print("Training Pitch Type Classifier")
        print("=" * 60)
        self.type_model = self._create_type_model(n_classes)
        self.type_model.fit(
            train_pool_type,
            eval_set=val_pool_type,
            use_best_model=True,
        )
        results["type_best_iteration"] = self.type_model.get_best_iteration()

        print("\n" + "=" * 60)
        print("Training Horizontal Location (px) Regressor")
        print("=" * 60)
        train_pool_px = Pool(X_train, y_px_train, cat_features=cat_features)
        val_pool_px = Pool(X_val, y_px_val, cat_features=cat_features)
        self.px_model = self._create_location_model()
        self.px_model.fit(
            train_pool_px,
            eval_set=val_pool_px,
            use_best_model=True,
        )
        results["px_best_iteration"] = self.px_model.get_best_iteration()

        print("\n" + "=" * 60)
        print("Training Vertical Location (pz) Regressor")
        print("=" * 60)
        train_pool_pz = Pool(X_train, y_pz_train, cat_features=cat_features)
        val_pool_pz = Pool(X_val, y_pz_val, cat_features=cat_features)
        self.pz_model = self._create_location_model()
        self.pz_model.fit(
            train_pool_pz,
            eval_set=val_pool_pz,
            use_best_model=True,
        )
        results["pz_best_iteration"] = self.pz_model.get_best_iteration()

        return results

    def predict(
        self,
        X,
        cat_features: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pool = Pool(X, cat_features=cat_features)

        type_probs = self.type_model.predict_proba(pool)
        type_preds = np.argmax(type_probs, axis=1)
        px_preds = self.px_model.predict(pool)
        pz_preds = self.pz_model.predict(pool)

        return type_probs, type_preds, px_preds, pz_preds

    def evaluate(
        self,
        X,
        y_type: np.ndarray,
        y_px: np.ndarray,
        y_pz: np.ndarray,
        cat_features: list[int],
    ) -> dict:
        type_probs, type_preds, px_preds, pz_preds = self.predict(X, cat_features)

        accuracy = accuracy_score(y_type, type_preds)
        f1_macro = f1_score(y_type, type_preds, average="macro", zero_division=0)
        f1_weighted = f1_score(
            y_type, type_preds, average="weighted", zero_division=0
        )
        top3_acc = top_k_accuracy_score(
            y_type, type_probs, k=3, labels=range(type_probs.shape[1])
        )
        mae_px = mean_absolute_error(y_px, px_preds)
        mae_pz = mean_absolute_error(y_pz, pz_preds)
        rmse_px = np.sqrt(mean_squared_error(y_px, px_preds))
        rmse_pz = np.sqrt(mean_squared_error(y_pz, pz_preds))
        euclidean = np.mean(
            np.sqrt((y_px - px_preds) ** 2 + (y_pz - pz_preds) ** 2)
        )

        return {
            "accuracy": accuracy,
            "top3_accuracy": top3_acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "mae_px": mae_px,
            "mae_pz": mae_pz,
            "rmse_px": rmse_px,
            "rmse_pz": rmse_pz,
            "euclidean_error": euclidean,
            "type_preds": type_preds,
            "type_probs": type_probs,
            "px_preds": px_preds,
            "pz_preds": pz_preds,
        }

    def get_feature_importance(self, top_n: int = 20) -> dict:
        result = {}

        if self.type_model is not None:
            importance = self.type_model.get_feature_importance()
            indices = np.argsort(importance)[::-1][:top_n]
            result["pitch_type"] = [
                (self.feature_columns[i], importance[i]) for i in indices
            ]

        if self.px_model is not None:
            importance = self.px_model.get_feature_importance()
            indices = np.argsort(importance)[::-1][:top_n]
            result["px"] = [
                (self.feature_columns[i], importance[i]) for i in indices
            ]

        if self.pz_model is not None:
            importance = self.pz_model.get_feature_importance()
            indices = np.argsort(importance)[::-1][:top_n]
            result["pz"] = [
                (self.feature_columns[i], importance[i]) for i in indices
            ]

        return result

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.type_model is not None:
            self.type_model.save_model(output_dir / "type_model.cbm")
        if self.px_model is not None:
            self.px_model.save_model(output_dir / "px_model.cbm")
        if self.pz_model is not None:
            self.pz_model.save_model(output_dir / "pz_model.cbm")

        info = {
            "feature_columns": self.feature_columns,
            "categorical_features": self.categorical_features,
        }
        (output_dir / "feature_info.json").write_text(
            __import__("json").dumps(info, indent=2)
        )

    @classmethod
    def load(cls, output_dir: str | Path) -> "PitchCatBoostModel":
        import json

        output_dir = Path(output_dir)
        model = cls()

        model.type_model = CatBoostClassifier()
        model.type_model.load_model(output_dir / "type_model.cbm")
        model.px_model = CatBoostRegressor()
        model.px_model.load_model(output_dir / "px_model.cbm")
        model.pz_model = CatBoostRegressor()
        model.pz_model.load_model(output_dir / "pz_model.cbm")

        info = json.loads((output_dir / "feature_info.json").read_text())
        model.feature_columns = info["feature_columns"]
        model.categorical_features = info["categorical_features"]
        return model


class PitchXGBoostModel(PitchCatBoostModel):
    """XGBoost-compatible tree baseline using one-hot encoded categoricals."""

    def __init__(
        self,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        depth: int = 8,
        early_stopping_rounds: int = 50,
        random_seed: int = 42,
        verbose: int = 100,
    ):
        super().__init__(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=3.0,
            early_stopping_rounds=early_stopping_rounds,
            task_type="CPU",
            random_seed=random_seed,
            verbose=verbose,
        )
        self._encoded_columns: list[str] = []

    def _require_xgboost(self):
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise RuntimeError(
                "xgboost is not installed. Add it to the environment before "
                "running the XGBoost experiment."
            ) from exc
        return XGBClassifier, XGBRegressor

    def _encode_features(self, X):
        import pandas as pd

        encoded = pd.get_dummies(
            X,
            columns=self.categorical_features,
            dummy_na=True,
        )
        if not self._encoded_columns:
            self._encoded_columns = list(encoded.columns)
            return encoded
        return encoded.reindex(columns=self._encoded_columns, fill_value=0)

    def _create_type_model(self, n_classes: int):
        XGBClassifier, _ = self._require_xgboost()
        return XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            n_estimators=self.iterations,
            learning_rate=self.learning_rate,
            max_depth=self.depth,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.random_seed,
            tree_method="hist",
            eval_metric="mlogloss",
            verbosity=1 if self.verbose else 0,
            early_stopping_rounds=self.early_stopping_rounds,
        )

    def _create_location_model(self):
        _, XGBRegressor = self._require_xgboost()
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=self.iterations,
            learning_rate=self.learning_rate,
            max_depth=self.depth,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.random_seed,
            tree_method="hist",
            eval_metric="rmse",
            verbosity=1 if self.verbose else 0,
            early_stopping_rounds=self.early_stopping_rounds,
        )

    def train(
        self,
        X_train,
        y_type_train: np.ndarray,
        y_px_train: np.ndarray,
        y_pz_train: np.ndarray,
        X_val,
        y_type_val: np.ndarray,
        y_px_val: np.ndarray,
        y_pz_val: np.ndarray,
        cat_features: list[int],
        n_classes: int = 11,
    ) -> dict:
        del cat_features
        results = {}

        X_train_enc = self._encode_features(X_train)
        X_val_enc = self._encode_features(X_val)

        print("\n" + "=" * 60)
        print("Training XGBoost Pitch Type Classifier")
        print("=" * 60)
        self.type_model = self._create_type_model(n_classes)
        self.type_model.fit(
            X_train_enc,
            y_type_train,
            eval_set=[(X_val_enc, y_type_val)],
            verbose=bool(self.verbose),
        )
        results["type_best_iteration"] = getattr(self.type_model, "best_iteration", None)

        print("\n" + "=" * 60)
        print("Training XGBoost Horizontal Location (px) Regressor")
        print("=" * 60)
        self.px_model = self._create_location_model()
        self.px_model.fit(
            X_train_enc,
            y_px_train,
            eval_set=[(X_val_enc, y_px_val)],
            verbose=bool(self.verbose),
        )
        results["px_best_iteration"] = getattr(self.px_model, "best_iteration", None)

        print("\n" + "=" * 60)
        print("Training XGBoost Vertical Location (pz) Regressor")
        print("=" * 60)
        self.pz_model = self._create_location_model()
        self.pz_model.fit(
            X_train_enc,
            y_pz_train,
            eval_set=[(X_val_enc, y_pz_val)],
            verbose=bool(self.verbose),
        )
        results["pz_best_iteration"] = getattr(self.pz_model, "best_iteration", None)
        return results

    def predict(
        self,
        X,
        cat_features: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        del cat_features
        X_enc = self._encode_features(X)
        type_probs = self.type_model.predict_proba(X_enc)
        type_preds = np.argmax(type_probs, axis=1)
        px_preds = self.px_model.predict(X_enc)
        pz_preds = self.pz_model.predict(X_enc)
        return type_probs, type_preds, px_preds, pz_preds

    def get_feature_importance(self, top_n: int = 20) -> dict:
        result = {}

        if self.type_model is not None and hasattr(self.type_model, "feature_importances_"):
            importance = np.asarray(self.type_model.feature_importances_)
            indices = np.argsort(importance)[::-1][:top_n]
            result["pitch_type"] = [
                (self._encoded_columns[i], float(importance[i])) for i in indices
            ]

        if self.px_model is not None and hasattr(self.px_model, "feature_importances_"):
            importance = np.asarray(self.px_model.feature_importances_)
            indices = np.argsort(importance)[::-1][:top_n]
            result["px"] = [
                (self._encoded_columns[i], float(importance[i])) for i in indices
            ]

        if self.pz_model is not None and hasattr(self.pz_model, "feature_importances_"):
            importance = np.asarray(self.pz_model.feature_importances_)
            indices = np.argsort(importance)[::-1][:top_n]
            result["pz"] = [
                (self._encoded_columns[i], float(importance[i])) for i in indices
            ]

        return result
