from src.ml.season_splits import (
    default_data_source_train_seasons,
    default_train_seasons,
    discover_available_seasons,
)


def test_default_train_seasons_uses_full_pre_validation_history():
    available = [str(season) for season in range(2009, 2027)]

    assert default_train_seasons(available, val_season="2024", test_season="2025") == [
        "2009",
        "2010",
        "2011",
        "2012",
        "2013",
        "2014",
        "2015",
        "2016",
        "2017",
        "2018",
        "2019",
        "2021",
        "2022",
        "2023",
    ]



def test_default_train_seasons_can_keep_2020_when_requested():
    available = ["2019", "2020", "2021", "2022", "2023", "2024", "2025"]

    assert default_train_seasons(
        available,
        val_season="2024",
        test_season="2025",
        exclude_2020=False,
    ) == ["2019", "2020", "2021", "2022", "2023"]



def test_discover_available_seasons_reads_local_parquet_tree(tmp_path):
    for season in ["2025", "2018", "2021"]:
        (tmp_path / season).mkdir()
    (tmp_path / "notes").mkdir()
    (tmp_path / ".cache").mkdir()

    assert discover_available_seasons(tmp_path) == ["2018", "2021", "2025"]



def test_default_data_source_train_seasons_uses_postgres_discovery(monkeypatch):
    monkeypatch.setattr(
        "src.ml.season_splits.discover_postgres_seasons",
        lambda: ["2009", "2010", "2020", "2024", "2025"],
    )

    assert default_data_source_train_seasons("postgres") == ["2009", "2010"]
