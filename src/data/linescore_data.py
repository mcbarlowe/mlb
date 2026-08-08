from pathlib import Path

import pandas as pd


class LinescoreData:
    """
    Data transformation class for MLB Linescore data.

    Transforms linescore JSON into inning-by-inning tabular format.
    """

    def transform(self, data: dict, game_pk: int = None) -> pd.DataFrame:
        """
        Transform linescore JSON into DataFrame.

        Args:
            data (dict): Raw GUMBO JSON response
            game_pk (int, optional): Game identifier

        Returns:
            pd.DataFrame: Inning-by-inning linescore data
        """
        linescore = data.get("liveData", {}).get("linescore", {})
        innings = linescore.get("innings", [])

        rows = []
        for inning in innings:
            inning_num = inning.get("num")
            ordinal_num = inning.get("ordinalNum")

            # Away team inning data
            if "away" in inning:
                away_row = {
                    "game_pk": game_pk,
                    "inning": inning_num,
                    "inning_ordinal": ordinal_num,
                    "team_type": "away",
                    "runs": inning["away"].get("runs"),
                    "hits": inning["away"].get("hits"),
                    "errors": inning["away"].get("errors"),
                    "left_on_base": inning["away"].get("leftOnBase"),
                }
                rows.append(away_row)

            # Home team inning data
            if "home" in inning:
                home_row = {
                    "game_pk": game_pk,
                    "inning": inning_num,
                    "inning_ordinal": ordinal_num,
                    "team_type": "home",
                    "runs": inning["home"].get("runs"),
                    "hits": inning["home"].get("hits"),
                    "errors": inning["home"].get("errors"),
                    "left_on_base": inning["home"].get("leftOnBase"),
                }
                rows.append(home_row)

        df = pd.DataFrame(rows)

        # Add linescore metadata
        if not df.empty and game_pk:
            df["current_inning"] = linescore.get("currentInning")
            df["inning_state"] = linescore.get("inningState")
            df["inning_half"] = linescore.get("inningHalf")
            df["scheduled_innings"] = linescore.get("scheduledInnings")

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

    def save_to_db(self, df: pd.DataFrame, db_handler, if_exists: str = "append") -> None:
        """
        Save DataFrame to the configured PostgreSQL database.

        Args:
            df (pd.DataFrame): DataFrame to save
            db_handler: PostgresHandler instance
            if_exists (str): How to behave if table exists ('append', 'replace', 'fail')
        """
        db_handler.insert_dataframe(df, "linescore", if_exists=if_exists)
