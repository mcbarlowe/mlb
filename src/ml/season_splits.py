from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from src.ml.postgres_data import discover_postgres_seasons

DEFAULT_VAL_SEASON = "2024"
DEFAULT_TEST_SEASON = "2025"



def normalize_seasons(seasons: Sequence[str | int]) -> list[str]:
    """Normalize season values to sorted four-digit strings."""
    return [str(season) for season in sorted({int(season) for season in seasons})]



def discover_available_seasons(data_path: str | Path) -> list[str]:
    """Discover available seasons from PostgreSQL or a local parquet tree."""
    if str(data_path) == "postgres":
        return normalize_seasons(discover_postgres_seasons())

    root = Path(data_path)
    if not root.exists():
        return []

    seasons = [
        child.name
        for child in root.iterdir()
        if child.is_dir() and child.name.isdigit()
    ]
    return normalize_seasons(seasons)



def default_train_seasons(
    available_seasons: Sequence[str | int],
    *,
    val_season: str = DEFAULT_VAL_SEASON,
    test_season: str = DEFAULT_TEST_SEASON,
    exclude_2020: bool = True,
) -> list[str]:
    """Build the default training window from available seasons.

    The training window includes every season strictly before the earlier of
    the validation and test seasons. The validation season, the test season,
    and (by default) the anomalous 2020 season are excluded.
    """
    normalized = [int(season) for season in normalize_seasons(available_seasons)]
    cutoff = min(int(val_season), int(test_season))
    excluded = {int(val_season), int(test_season)}
    if exclude_2020:
        excluded.add(2020)

    return [str(season) for season in normalized if season < cutoff and season not in excluded]



def default_data_source_train_seasons(
    data_path: str | Path,
    *,
    val_season: str = DEFAULT_VAL_SEASON,
    test_season: str = DEFAULT_TEST_SEASON,
    exclude_2020: bool = True,
) -> list[str]:
    """Resolve the default training seasons for a configured data source."""
    return default_train_seasons(
        discover_available_seasons(data_path),
        val_season=val_season,
        test_season=test_season,
        exclude_2020=exclude_2020,
    )
