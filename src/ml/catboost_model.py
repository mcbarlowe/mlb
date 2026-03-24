"""
CatBoost model for pitch prediction.

This module provides a gradient boosting approach as an alternative to the LSTM model.
CatBoost handles categorical features natively and often excels on tabular data.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    top_k_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
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
        """
        Initialize CatBoost models.

        Args:
            iterations: Maximum number of boosting iterations.
            learning_rate: Learning rate for gradient boosting.
            depth: Depth of trees.
            l2_leaf_reg: L2 regularization coefficient.
            early_stopping_rounds: Rounds for early stopping.
            task_type: "CPU" or "GPU".
            random_seed: Random seed for reproducibility.
            verbose: Logging frequency (0 for silent).
        """
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.early_stopping_rounds = early_stopping_rounds
        self.task_type = task_type
        self.random_seed = random_seed
        self.verbose = verbose

        # Models will be created during training
        self.type_model: Optional[CatBoostClassifier] = None
        self.px_model: Optional[CatBoostRegressor] = None
        self.pz_model: Optional[CatBoostRegressor] = None

        # Feature information
        self.feature_columns: list[str] = []
        self.categorical_features: list[str] = []

    def _create_type_model(self, n_classes: int) -> CatBoostClassifier:
        """Create pitch type classifier."""
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
        """Create location regressor."""
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
        """
        Get feature columns for CatBoost.

        Uses raw IDs for categorical features instead of encoded indices,
        since CatBoost handles categoricals natively.
        """
        return [
            # Count features
            "balls",
            "strikes",
            "two_strike_count",
            "hitters_count",
            "first_pitch",
            "pitcher_ahead",
            # Game state
            "inning",
            "outs",
            "runners_bitmap",
            "score_diff",
            # Categorical features (CatBoost handles these natively)
            "pitcher_id",
            "batter_id",
            # Handedness
            "throw_side",
            "bat_side",
            "platoon_same_side",
            # Pitcher tendencies
            "pitcher_ff_pct",
            "pitcher_repertoire",
            # Batter zone
            "batter_zone_height",
            "batter_zone_mid",
            # Previous pitch (categorical)
            "prev_pitch_type",
            # Previous pitch features
            "prev_px",
            "prev_pz",
            "prev_speed",
            "prev_is_strike",
            # Velocity
            "velocity_delta",
            # Swing/result
            "prev_swing",
            "prev_result_type",
            # Cumulative
            "n_fastballs_in_ab",
            "n_breaking_in_ab",
            # Sequence
            "same_pitch_streak",
            "pitch_number",
            # Weather features (NEW)
            "temp_normalized",
            "wind_speed",
            "wind_direction",
            "is_night_game",
            # Temporal features (NEW)
            "season_progress",
            # Enhanced game situation (NEW)
            "runners_in_scoring_position",
            "leverage_approx",
            # Pitcher fatigue (NEW)
            "pitcher_pitch_count",
        ]

    def get_categorical_features(self) -> list[str]:
        """Get list of categorical feature names."""
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
        """
        Prepare data for CatBoost training.

        CatBoost works with flat tabular data, not sequences.
        Each pitch is a separate sample.

        Args:
            df: Raw DataFrame with pitch data.
            feature_engine: Fitted PitchFeatureEngine for accessing pitcher stats.

        Returns:
            Tuple of (X, y_type, y_px, y_pz, cat_feature_indices)
        """
        from src.ml.features import PITCH_TYPE_TO_IDX

        # Transform data to get engineered features
        df = feature_engine.transform(df)

        # Add previous pitch type as string (for CatBoost categorical)
        df = df.with_columns([
            pl.col("pitch_type_code")
            .shift(1)
            .over(["game_pk", "at_bat_index"])
            .fill_null("NONE")
            .alias("prev_pitch_type"),
        ])

        # Filter nulls in targets
        df = df.filter(
            pl.col("pitch_type_idx").is_not_null()
            & pl.col("px").is_not_null()
            & pl.col("pz").is_not_null()
        )

        # Get feature columns
        feature_cols = self.get_feature_columns()
        cat_cols = self.get_categorical_features()

        # Verify all columns exist
        available_cols = []
        for col in feature_cols:
            if col in df.columns:
                available_cols.append(col)
            else:
                print(f"Warning: Column {col} not found, skipping")

        self.feature_columns = available_cols

        # Get categorical feature indices
        cat_indices = [
            i for i, col in enumerate(available_cols)
            if col in cat_cols
        ]
        self.categorical_features = [available_cols[i] for i in cat_indices]

        # Extract features and targets
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
        """
        Train all three models (type, px, pz).

        Args:
            X_train, X_val: Feature DataFrames.
            y_type_train, y_type_val: Pitch type targets.
            y_px_train, y_px_val: Horizontal location targets.
            y_pz_train, y_pz_val: Vertical location targets.
            cat_features: Indices of categorical features.
            n_classes: Number of pitch type classes.

        Returns:
            Dictionary with training results.
        """
        results = {}

        # Create CatBoost Pools
        train_pool_type = Pool(
            X_train, y_type_train,
            cat_features=cat_features,
        )
        val_pool_type = Pool(
            X_val, y_type_val,
            cat_features=cat_features,
        )

        # Train pitch type classifier
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

        # Train px regressor
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

        # Train pz regressor
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
        """
        Make predictions.

        Args:
            X: Feature DataFrame.
            cat_features: Indices of categorical features.

        Returns:
            Tuple of (type_probs, type_preds, px_preds, pz_preds)
        """
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
        """
        Evaluate model on test data.

        Args:
            X: Feature DataFrame.
            y_type: True pitch types.
            y_px: True horizontal locations.
            y_pz: True vertical locations.
            cat_features: Indices of categorical features.

        Returns:
            Dictionary of evaluation metrics.
        """
        type_probs, type_preds, px_preds, pz_preds = self.predict(X, cat_features)

        # Classification metrics
        accuracy = accuracy_score(y_type, type_preds)
        f1_macro = f1_score(y_type, type_preds, average="macro", zero_division=0)
        f1_weighted = f1_score(y_type, type_preds, average="weighted", zero_division=0)

        # Top-3 accuracy
        top3_acc = top_k_accuracy_score(y_type, type_probs, k=3, labels=range(type_probs.shape[1]))

        # Location metrics
        mae_px = mean_absolute_error(y_px, px_preds)
        mae_pz = mean_absolute_error(y_pz, pz_preds)
        rmse_px = np.sqrt(mean_squared_error(y_px, px_preds))
        rmse_pz = np.sqrt(mean_squared_error(y_pz, pz_preds))

        # Euclidean error
        euclidean = np.mean(np.sqrt((y_px - px_preds) ** 2 + (y_pz - pz_preds) ** 2))

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
        """
        Get feature importance for all models.

        Args:
            top_n: Number of top features to return.

        Returns:
            Dictionary with feature importance for each model.
        """
        result = {}

        if self.type_model is not None:
            importance = self.type_model.get_feature_importance()
            indices = np.argsort(importance)[::-1][:top_n]
            result["pitch_type"] = [
                (self.feature_columns[i], importance[i])
                for i in indices
            ]

        if self.px_model is not None:
            importance = self.px_model.get_feature_importance()
            indices = np.argsort(importance)[::-1][:top_n]
            result["px"] = [
                (self.feature_columns[i], importance[i])
                for i in indices
            ]

        if self.pz_model is not None:
            importance = self.pz_model.get_feature_importance()
            indices = np.argsort(importance)[::-1][:top_n]
            result["pz"] = [
                (self.feature_columns[i], importance[i])
                for i in indices
            ]

        return result

    def save(self, path: Path) -> None:
        """Save all models to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.type_model is not None:
            self.type_model.save_model(str(path / "type_model.cbm"))
        if self.px_model is not None:
            self.px_model.save_model(str(path / "px_model.cbm"))
        if self.pz_model is not None:
            self.pz_model.save_model(str(path / "pz_model.cbm"))

        # Save feature info
        import json
        with open(path / "feature_info.json", "w") as f:
            json.dump({
                "feature_columns": self.feature_columns,
                "categorical_features": self.categorical_features,
            }, f)

    def load(self, path: Path) -> None:
        """Load all models from directory."""
        path = Path(path)

        self.type_model = CatBoostClassifier()
        self.type_model.load_model(str(path / "type_model.cbm"))

        self.px_model = CatBoostRegressor()
        self.px_model.load_model(str(path / "px_model.cbm"))

        self.pz_model = CatBoostRegressor()
        self.pz_model.load_model(str(path / "pz_model.cbm"))

        # Load feature info
        import json
        with open(path / "feature_info.json") as f:
            info = json.load(f)
            self.feature_columns = info["feature_columns"]
            self.categorical_features = info["categorical_features"]
