from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa


class DataEngine:
    """
    DuckDB-based analytical data engine for NEXUS-6.

    The engine provides:
        - Direct querying of CSV, Parquet, and JSON files
        - In-memory table registration
        - Dataset metadata inspection
        - SQL execution
        - Arrow result support

    DuckDB performs the analytical work, while PyArrow provides a
    columnar representation for efficient data exchange.
    """

    SUPPORTED_FORMATS = {
        ".csv": "csv",
        ".parquet": "parquet",
        ".json": "json",
        ".jsonl": "json",
        ".ndjson": "json",
    }

    def __init__(
        self,
        database_path: str = ":memory:",
    ) -> None:
        """
        Create a DuckDB analytical engine.

        Parameters
        ----------
        database_path:
            DuckDB database location.

            The default ':memory:' creates an in-memory analytical
            database, which is appropriate for isolated analysis jobs.
        """

        self.database_path = database_path
        self.connection = duckdb.connect(database_path)

    def close(self) -> None:
        """Close the DuckDB connection."""

        if self.connection is not None:
            self.connection.close()

    def __enter__(self) -> "DataEngine":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()

    def register_arrow(
        self,
        name: str,
        table: pa.Table,
    ) -> None:
        """
        Register a PyArrow table as a DuckDB relation.

        The table becomes queryable using the supplied relation name.
        """

        self.connection.register(name, table)

    def register_dataframe(
        self,
        name: str,
        dataframe: Any,
    ) -> None:
        """
        Register a dataframe-like object with DuckDB.

        The object is intentionally typed as Any so the core engine does
        not require pandas to be imported at module load time.
        """

        self.connection.register(name, dataframe)

    def unregister(self, name: str) -> None:
        """Remove a registered relation."""

        self.connection.unregister(name)

    def execute(
        self,
        query: str,
    ) -> duckdb.DuckDBPyConnection:
        """
        Execute a SQL statement.

        Returns the DuckDB connection so callers can choose the desired
        result representation.
        """

        return self.connection.execute(query)

    def fetch_arrow(
        self,
        query: str,
    ) -> pa.Table:
        """
        Execute a query and return the result as a PyArrow table.
        """

        return self.connection.execute(query).fetch_arrow_table()

    def fetch_dicts(
        self,
        query: str,
    ) -> list[dict[str, Any]]:
        """
        Execute a query and return rows as dictionaries.

        This representation is convenient for API responses and
        structured observation objects.
        """

        result = self.connection.execute(query)

        column_names = [
            description[0]
            for description in result.description
        ]

        rows = result.fetchall()

        return [
            dict(zip(column_names, row))
            for row in rows
        ]

    def fetch_rows(
        self,
        query: str,
    ) -> list[tuple[Any, ...]]:
        """Execute a query and return ordinary Python tuples."""

        return self.connection.execute(query).fetchall()

    def read_file(
        self,
        file_path: str | Path,
        table_name: str = "dataset",
    ) -> None:
        """
        Register a supported dataset file as a DuckDB view.

        CSV, Parquet, and JSON-family files are queried directly by
        DuckDB rather than first being loaded into a pandas dataframe.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Dataset path is not a file: {path}"
            )

        extension = path.suffix.lower()

        file_format = self.SUPPORTED_FORMATS.get(extension)

        if file_format is None:
            supported = ", ".join(
                sorted(self.SUPPORTED_FORMATS.keys())
            )

            raise ValueError(
                f"Unsupported dataset format '{extension}'. "
                f"Supported formats: {supported}"
            )

        escaped_path = self._escape_sql_string(
            str(path.resolve())
        )

        if file_format == "csv":
            query = (
                f"CREATE OR REPLACE VIEW {table_name} AS "
                f"SELECT * FROM read_csv_auto('{escaped_path}')"
            )

        elif file_format == "parquet":
            query = (
                f"CREATE OR REPLACE VIEW {table_name} AS "
                f"SELECT * FROM read_parquet('{escaped_path}')"
            )

        else:
            query = (
                f"CREATE OR REPLACE VIEW {table_name} AS "
                f"SELECT * FROM read_json_auto('{escaped_path}')"
            )

        self.connection.execute(query)

    def inspect_schema(
        self,
        table_name: str = "dataset",
    ) -> list[dict[str, Any]]:
        """
        Return column-level schema metadata for a registered relation.
        """

        result = self.connection.execute(
            f"DESCRIBE {table_name}"
        )

        column_names = [
            description[0]
            for description in result.description
        ]

        rows = result.fetchall()

        return [
            dict(zip(column_names, row))
            for row in rows
        ]

    def inspect_metadata(
        self,
        table_name: str = "dataset",
    ) -> dict[str, Any]:
        """
        Return basic dataset metadata.

        Metadata is calculated by DuckDB and does not require loading the
        complete dataset into Python.
        """

        schema = self.inspect_schema(table_name)

        row_count = self.connection.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]

        return {
            "table_name": table_name,
            "row_count": row_count,
            "column_count": len(schema),
            "columns": schema,
        }

    def profile_numeric_columns(
        self,
        table_name: str = "dataset",
    ) -> list[dict[str, Any]]:
        """
        Produce basic statistics for numeric columns.

        The calculation is performed inside DuckDB.
        """

        numeric_columns = self.connection.execute(
            f"""
            SELECT column_name
            FROM (
                DESCRIBE {table_name}
            )
            WHERE
                LOWER(column_type) LIKE '%int%'
                OR LOWER(column_type) LIKE '%decimal%'
                OR LOWER(column_type) LIKE '%double%'
                OR LOWER(column_type) LIKE '%float%'
            """
        ).fetchall()

        results: list[dict[str, Any]] = []

        for row in numeric_columns:
            column_name = row[0]
            quoted_column = self._quote_identifier(column_name)

            statistics = self.connection.execute(
                f"""
                SELECT
                    COUNT({quoted_column}) AS non_null_count,
                    MIN({quoted_column}) AS minimum,
                    MAX({quoted_column}) AS maximum,
                    AVG({quoted_column}) AS mean,
                    STDDEV_SAMP({quoted_column}) AS standard_deviation
                FROM {table_name}
                """
            ).fetchone()

            results.append(
                {
                    "column": column_name,
                    "non_null_count": statistics[0],
                    "minimum": statistics[1],
                    "maximum": statistics[2],
                    "mean": statistics[3],
                    "standard_deviation": statistics[4],
                }
            )

        return results

    @staticmethod
    def _escape_sql_string(value: str) -> str:
        """Escape a value used inside a SQL string literal."""

        return value.replace("'", "''")

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        """Safely quote a SQL identifier."""

        escaped = identifier.replace('"', '""')

        return f'"{escaped}"'