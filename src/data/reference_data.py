from pathlib import Path

import pandas as pd


class ReferenceData:
    """
    Data transformation class for MLB reference endpoints.

    Transforms simple reference data (positions, pitch types, etc.) into tabular format.
    """

    def transform(self, data: dict, key_field: str = None) -> pd.DataFrame:
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
        # Handle different response structures
        if key_field and key_field in data:
            items = data[key_field]
        elif isinstance(data, list):
            items = data
        else:
            # Try to find the first list in the response
            for key, value in data.items():
                if isinstance(value, list):
                    items = value
                    break
            else:
                # If no list found, wrap the entire dict
                items = [data]

        df = pd.json_normalize(items)
        return df

    def save(self, df: pd.DataFrame, output_path: Path, format: str = "parquet") -> None:
        """
        Save DataFrame to file.

        Args:
            df (pd.DataFrame): DataFrame to save
            output_path (Path): Output file path
            format (str): Output format ('parquet', 'csv', or 'json')

        Raises:
            ValueError: If format is not supported
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            df.to_parquet(output_path, index=False)
        elif format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient="records", lines=True)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'parquet', 'csv', or 'json'.")

    def save_to_db(self, df: pd.DataFrame, table_name: str, db_handler,
                   if_exists: str = "append") -> None:
        """
        Save DataFrame to DuckDB database.

        Args:
            df (pd.DataFrame): DataFrame to save
            table_name (str): Table name (e.g., 'positions', 'pitch_types', 'venues')
            db_handler: DuckDBHandler instance
            if_exists (str): How to behave if table exists ('append', 'replace', 'fail')
        """
        db_handler.insert_dataframe(df, table_name, if_exists=if_exists)
