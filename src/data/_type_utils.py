import pandas as pd


def coerce_dataframe_types(df: pd.DataFrame, data_types: dict[str, type]) -> pd.DataFrame:
    """Coerce pandas columns using nullable-friendly dtypes for historical payloads."""

    coerced_df = df.copy()
    for column, expected_type in data_types.items():
        if column not in coerced_df.columns:
            continue

        series = coerced_df[column]
        if expected_type is int:
            coerced_df[column] = pd.to_numeric(series, errors="coerce").astype("Int64")
        elif expected_type is float:
            coerced_df[column] = pd.to_numeric(series, errors="coerce").astype(float)
        elif expected_type is bool:
            normalized = series.map(
                lambda value: value
                if pd.isna(value) or isinstance(value, bool)
                else {
                    "y": True,
                    "yes": True,
                    "true": True,
                    "t": True,
                    "1": True,
                    "n": False,
                    "no": False,
                    "false": False,
                    "f": False,
                    "0": False,
                }.get(str(value).strip().lower(), value)
            )
            coerced_df[column] = normalized.astype("boolean").fillna(False).astype(bool)
        elif expected_type is str:
            coerced_df[column] = series.where(series.notna(), None)
        else:
            coerced_df[column] = series.astype(expected_type)

    return coerced_df
