from pathlib import Path

import pandas as pd


class ReferenceData:
    """
    Data transformation class for MLB reference endpoints.

    Transforms simple reference data (positions, pitch types, etc.) into tabular format.
    """

    def transform(self, data: dict | list[dict], key_field: str | None = None) -> pd.DataFrame:
        """
        Transform reference data JSON into DataFrame.

        Args:
            data (dict): Raw API JSON response
            key_field (str, optional): Top-level key containing array data

        Returns:
            pd.DataFrame: Flattened reference data

        Examples:
            >>> # For positions endpoint
            >>> ref = ReferenceData()
            >>> df = ref.transform(positions_json)
            >>>
            >>> # For nested data
            >>> df = ref.transform(response_json, key_field='positions')
        """
        if isinstance(data, dict) and key_field and key_field in data:
            items = data[key_field]
        elif isinstance(data, list):
            items = data
        else:
            for value in data.values():
                if isinstance(value, list):
                    items = value
                    break
            else:
                items = [data]

        df = pd.json_normalize(items)
        return df

    def save(self, df: pd.DataFrame, output_path: Path, format: str = "parquet") -> None:
        """Save DataFrame to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            df.to_parquet(output_path, index=False)
        elif format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient="records", lines=True)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'parquet', 'csv', or 'json'.")

    def _normalize_for_table(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        if table_name == "positions":
            normalized_df = df.rename(
                columns={
                    "fullName": "name",
                    "abbrev": "abbreviation",
                }
            )
            required_columns = ["code", "name", "type", "abbreviation"]
            dedupe_columns = ["code"]
        elif table_name in {"pitch_types", "event_types"}:
            normalized_df = df
            required_columns = ["code", "description"]
            dedupe_columns = ["code"]
        elif table_name == "game_types":
            normalized_df = df
            required_columns = ["id", "description"]
            dedupe_columns = ["id"]
        else:
            return df

        missing_columns = [column for column in required_columns if column not in normalized_df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns for {table_name}: {missing_columns}. "
                f"Available columns: {normalized_df.columns.tolist()}"
            )

        return (
            normalized_df.loc[:, required_columns]
            .drop_duplicates(subset=dedupe_columns, keep="first")
            .reset_index(drop=True)
            .copy()
        )

    def save_to_db(self, df: pd.DataFrame, table_name: str, db_handler,
                   if_exists: str = "append") -> None:
        """
        Save DataFrame to the configured PostgreSQL database.

        Args:
            df (pd.DataFrame): DataFrame to save
            table_name (str): Table name (e.g., 'positions', 'pitch_types', 'venues')
            db_handler: PostgresHandler instance
            if_exists (str): How to behave if table exists ('append', 'replace', 'fail')
        """
        normalized_df = self._normalize_for_table(df, table_name)
        db_handler.insert_dataframe(normalized_df, table_name, if_exists=if_exists)
