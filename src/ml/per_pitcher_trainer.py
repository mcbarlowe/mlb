"""
Per-pitcher model training for pitch type prediction.

Research shows that pitcher-specific models can achieve ~76.7% accuracy
compared to ~66% for a single global model. This module implements
training and inference for per-pitcher CatBoost models.
"""

import json
from pathlib import Path
from typing import Optional
import numpy as np
import polars as pl
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from tqdm import tqdm


class PerPitcherTrainer:
    """
    Trains individual CatBoost models for high-volume pitchers.

    Falls back to a global model for pitchers without enough data.
    """

    def __init__(
        self,
        min_pitches: int = 3000,
        iterations: int = 200,
        learning_rate: float = 0.1,
        depth: int = 6,
        early_stopping_rounds: int = 20,
        random_seed: int = 42,
        verbose: int = 0,
    ):
        """
        Initialize per-pitcher trainer.

        Args:
            min_pitches: Minimum pitches required to train a pitcher-specific model.
            iterations: Max boosting iterations per model.
            learning_rate: Learning rate (higher since less data per model).
            depth: Tree depth (lower since less data).
            early_stopping_rounds: Early stopping patience.
            random_seed: Random seed for reproducibility.
            verbose: CatBoost verbosity (0 for silent).
        """
        self.min_pitches = min_pitches
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.early_stopping_rounds = early_stopping_rounds
        self.random_seed = random_seed
        self.verbose = verbose

        # Storage for trained models
        self.pitcher_models: dict[int, CatBoostClassifier] = {}
        self.global_model: Optional[CatBoostClassifier] = None
        self.pitcher_stats: dict[int, dict] = {}

        # Feature configuration
        self.feature_columns: list[str] = []
        self.categorical_features: list[str] = []
        self.n_classes: int = 11

    def get_feature_columns(self) -> list[str]:
        """
        Get feature columns for per-pitcher models.

        Excludes pitcher_id since we're training per-pitcher.
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
            # Batter (kept since it varies)
            "batter_id",
            # Handedness
            "throw_side",
            "bat_side",
            "platoon_same_side",
            # Pitcher tendencies (still useful even per-pitcher)
            "pitcher_ff_pct",
            "pitcher_repertoire",
            # Batter zone
            "batter_zone_height",
            "batter_zone_mid",
            # Previous pitch
            "prev_pitch_type",
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
            # Weather features
            "temp_normalized",
            "wind_speed",
            "wind_direction",
            "is_night_game",
            # Temporal
            "season_progress",
            # Game situation
            "runners_in_scoring_position",
            "leverage_approx",
            # Pitcher fatigue
            "pitcher_pitch_count",
        ]

    def get_categorical_features(self) -> list[str]:
        """Get categorical feature names."""
        return [
            "batter_id",
            "throw_side",
            "bat_side",
            "prev_pitch_type",
        ]

    def _create_model(self, n_classes: int) -> CatBoostClassifier:
        """Create a CatBoost classifier for a single pitcher."""
        return CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=3.0,
            loss_function="MultiClass",
            classes_count=n_classes,
            task_type="CPU",
            random_seed=self.random_seed,
            verbose=self.verbose,
            early_stopping_rounds=self.early_stopping_rounds,
            thread_count=-1,
        )

    def prepare_data(
        self,
        df: pl.DataFrame,
        feature_engine,
    ) -> pl.DataFrame:
        """
        Prepare data with features for training.

        Args:
            df: Raw DataFrame with pitch data.
            feature_engine: Fitted PitchFeatureEngine.

        Returns:
            DataFrame with engineered features.
        """
        # Transform data
        df = feature_engine.transform(df)

        # Add previous pitch type as string for CatBoost categorical
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

        return df

    def train(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        feature_engine,
        pitcher_counts: Optional[pl.DataFrame] = None,
        max_pitchers: Optional[int] = None,
    ) -> dict:
        """
        Train per-pitcher models and a global fallback model.

        Args:
            train_df: Training DataFrame (raw).
            val_df: Validation DataFrame (raw).
            feature_engine: Fitted PitchFeatureEngine.
            pitcher_counts: DataFrame with pitcher_id and pitch counts.
            max_pitchers: Maximum number of pitcher models to train (for testing).

        Returns:
            Dictionary with training results.
        """
        print("Preparing training data...")
        train_df = self.prepare_data(train_df, feature_engine)
        val_df = self.prepare_data(val_df, feature_engine)

        # Get feature columns
        feature_cols = self.get_feature_columns()
        cat_cols = self.get_categorical_features()

        # Filter to available columns
        available_cols = [c for c in feature_cols if c in train_df.columns]
        self.feature_columns = available_cols

        # Get categorical indices
        cat_indices = [i for i, c in enumerate(available_cols) if c in cat_cols]
        self.categorical_features = [available_cols[i] for i in cat_indices]

        print(f"Using {len(available_cols)} features, {len(cat_indices)} categorical")

        # Get unique pitch types
        self.n_classes = train_df["pitch_type_idx"].max() + 1

        # Identify high-volume pitchers
        if pitcher_counts is None:
            pitcher_counts = (
                train_df
                .group_by("pitcher_id")
                .len()
                .sort("len", descending=True)
            )

        eligible_pitchers = pitcher_counts.filter(
            pl.col("len") >= self.min_pitches
        )["pitcher_id"].to_list()

        if max_pitchers:
            eligible_pitchers = eligible_pitchers[:max_pitchers]

        print(f"\nTraining models for {len(eligible_pitchers)} pitchers with >={self.min_pitches} pitches")

        results = {
            "n_pitcher_models": 0,
            "pitcher_accuracies": [],
            "global_accuracy": 0.0,
        }

        # Train per-pitcher models
        for pitcher_id in tqdm(eligible_pitchers, desc="Training pitcher models"):
            # Filter data for this pitcher
            p_train = train_df.filter(pl.col("pitcher_id") == pitcher_id)
            p_val = val_df.filter(pl.col("pitcher_id") == pitcher_id)

            if len(p_train) < 100 or len(p_val) < 20:
                continue

            # Extract features and targets
            X_train = p_train.select(available_cols).to_pandas()
            y_train = p_train["pitch_type_idx"].to_numpy()
            X_val = p_val.select(available_cols).to_pandas()
            y_val = p_val["pitch_type_idx"].to_numpy()

            # Create pools
            train_pool = Pool(X_train, y_train, cat_features=cat_indices)
            val_pool = Pool(X_val, y_val, cat_features=cat_indices)

            # Train model
            model = self._create_model(self.n_classes)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)

            # Evaluate
            y_pred = model.predict(val_pool).flatten()
            accuracy = accuracy_score(y_val, y_pred)

            # Store model and stats
            self.pitcher_models[pitcher_id] = model
            self.pitcher_stats[pitcher_id] = {
                "train_samples": len(p_train),
                "val_samples": len(p_val),
                "accuracy": accuracy,
                "best_iteration": model.get_best_iteration(),
            }

            results["pitcher_accuracies"].append(accuracy)
            results["n_pitcher_models"] += 1

        # Train global fallback model on remaining pitchers
        print("\nTraining global fallback model...")
        other_pitchers = set(train_df["pitcher_id"].unique().to_list()) - set(eligible_pitchers)

        # Include pitcher_id for global model
        global_features = ["pitcher_id"] + available_cols
        global_cat_indices = [0] + [i + 1 for i in cat_indices]  # Shift indices for pitcher_id

        g_train = train_df.filter(pl.col("pitcher_id").is_in(list(other_pitchers)))
        g_val = val_df.filter(pl.col("pitcher_id").is_in(list(other_pitchers)))

        if len(g_train) > 0 and len(g_val) > 0:
            X_train = g_train.select(global_features).to_pandas()
            y_train = g_train["pitch_type_idx"].to_numpy()
            X_val = g_val.select(global_features).to_pandas()
            y_val = g_val["pitch_type_idx"].to_numpy()

            train_pool = Pool(X_train, y_train, cat_features=global_cat_indices)
            val_pool = Pool(X_val, y_val, cat_features=global_cat_indices)

            # Use same iterations as pitcher models for consistency
            self.global_model = CatBoostClassifier(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                l2_leaf_reg=3.0,
                loss_function="MultiClass",
                classes_count=self.n_classes,
                task_type="CPU",
                random_seed=self.random_seed,
                verbose=100,
                early_stopping_rounds=self.early_stopping_rounds,
            )
            self.global_model.fit(train_pool, eval_set=val_pool, use_best_model=True)

            y_pred = self.global_model.predict(val_pool).flatten()
            results["global_accuracy"] = accuracy_score(y_val, y_pred)

        # Summary stats
        if results["pitcher_accuracies"]:
            results["mean_pitcher_accuracy"] = np.mean(results["pitcher_accuracies"])
            results["median_pitcher_accuracy"] = np.median(results["pitcher_accuracies"])
            results["min_pitcher_accuracy"] = np.min(results["pitcher_accuracies"])
            results["max_pitcher_accuracy"] = np.max(results["pitcher_accuracies"])

        return results

    def predict(
        self,
        df: pl.DataFrame,
        feature_engine,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions using appropriate models.

        Args:
            df: DataFrame with pitch data (raw).
            feature_engine: Fitted PitchFeatureEngine.

        Returns:
            Tuple of (predictions, probabilities, model_used) where model_used
            indicates 1=pitcher-specific, 0=global.
        """
        df = self.prepare_data(df, feature_engine)

        predictions = np.zeros(len(df), dtype=np.int64)
        probabilities = np.zeros((len(df), self.n_classes), dtype=np.float32)
        model_used = np.zeros(len(df), dtype=np.int8)

        # Get feature columns
        available_cols = self.feature_columns
        cat_indices = [i for i, c in enumerate(available_cols) if c in self.categorical_features]

        # Global model features (includes pitcher_id)
        global_features = ["pitcher_id"] + available_cols
        global_cat_indices = [0] + [i + 1 for i in cat_indices]

        # Group by pitcher
        for pitcher_id in df["pitcher_id"].unique().to_list():
            mask = (df["pitcher_id"] == pitcher_id).to_numpy()
            p_df = df.filter(pl.col("pitcher_id") == pitcher_id)

            if pitcher_id in self.pitcher_models:
                # Use pitcher-specific model
                X = p_df.select(available_cols).to_pandas()
                pool = Pool(X, cat_features=cat_indices)

                probs = self.pitcher_models[pitcher_id].predict_proba(pool)
                preds = np.argmax(probs, axis=1)

                predictions[mask] = preds
                probabilities[mask] = probs
                model_used[mask] = 1
            elif self.global_model is not None:
                # Use global fallback
                X = p_df.select(global_features).to_pandas()
                pool = Pool(X, cat_features=global_cat_indices)

                probs = self.global_model.predict_proba(pool)
                preds = np.argmax(probs, axis=1)

                predictions[mask] = preds
                probabilities[mask] = probs
                model_used[mask] = 0

        return predictions, probabilities, model_used

    def evaluate(
        self,
        df: pl.DataFrame,
        feature_engine,
    ) -> dict:
        """
        Evaluate on test data.

        Args:
            df: Test DataFrame (raw).
            feature_engine: Fitted PitchFeatureEngine.

        Returns:
            Dictionary with evaluation metrics.
        """
        df_prepared = self.prepare_data(df, feature_engine)
        y_true = df_prepared["pitch_type_idx"].to_numpy()

        predictions, probabilities, model_used = self.predict(df, feature_engine)

        # Overall metrics
        accuracy = accuracy_score(y_true, predictions)
        f1_macro = f1_score(y_true, predictions, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, predictions, average="weighted", zero_division=0)
        top3_acc = top_k_accuracy_score(y_true, probabilities, k=3, labels=range(self.n_classes))

        # Metrics by model type
        pitcher_mask = model_used == 1
        global_mask = model_used == 0

        results = {
            "overall_accuracy": accuracy,
            "overall_top3_accuracy": top3_acc,
            "overall_f1_macro": f1_macro,
            "overall_f1_weighted": f1_weighted,
            "n_samples": len(y_true),
            "n_pitcher_model_samples": int(pitcher_mask.sum()),
            "n_global_model_samples": int(global_mask.sum()),
        }

        if pitcher_mask.sum() > 0:
            results["pitcher_model_accuracy"] = accuracy_score(
                y_true[pitcher_mask], predictions[pitcher_mask]
            )

        if global_mask.sum() > 0:
            results["global_model_accuracy"] = accuracy_score(
                y_true[global_mask], predictions[global_mask]
            )

        return results

    def save(self, path: Path) -> None:
        """Save all models and configuration."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save pitcher models
        pitcher_dir = path / "pitcher_models"
        pitcher_dir.mkdir(exist_ok=True)

        for pitcher_id, model in self.pitcher_models.items():
            model.save_model(str(pitcher_dir / f"{pitcher_id}.cbm"))

        # Save global model
        if self.global_model is not None:
            self.global_model.save_model(str(path / "global_model.cbm"))

        # Save configuration
        config = {
            "min_pitches": self.min_pitches,
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "feature_columns": self.feature_columns,
            "categorical_features": self.categorical_features,
            "n_classes": self.n_classes,
            "pitcher_ids": list(self.pitcher_models.keys()),
            "pitcher_stats": {str(k): v for k, v in self.pitcher_stats.items()},
        }

        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved {len(self.pitcher_models)} pitcher models to {path}")

    def load(self, path: Path) -> None:
        """Load all models and configuration."""
        path = Path(path)

        # Load configuration
        with open(path / "config.json") as f:
            config = json.load(f)

        self.min_pitches = config["min_pitches"]
        self.iterations = config["iterations"]
        self.learning_rate = config["learning_rate"]
        self.depth = config["depth"]
        self.feature_columns = config["feature_columns"]
        self.categorical_features = config["categorical_features"]
        self.n_classes = config["n_classes"]
        self.pitcher_stats = {int(k): v for k, v in config["pitcher_stats"].items()}

        # Load pitcher models
        pitcher_dir = path / "pitcher_models"
        for pitcher_id in config["pitcher_ids"]:
            model = CatBoostClassifier()
            model.load_model(str(pitcher_dir / f"{pitcher_id}.cbm"))
            self.pitcher_models[pitcher_id] = model

        # Load global model
        global_path = path / "global_model.cbm"
        if global_path.exists():
            self.global_model = CatBoostClassifier()
            self.global_model.load_model(str(global_path))

        print(f"Loaded {len(self.pitcher_models)} pitcher models from {path}")
