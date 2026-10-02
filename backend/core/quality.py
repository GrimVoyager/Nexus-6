from __future__ import annotations

from typing import Any

from backend.core.loader import DataEngine


class DataQualityAnalyzer:
    """
    Performs deterministic data-quality analysis using DuckDB.

    The analyzer checks:
        - Missing values
        - Missing-value percentages
        - Duplicate rows
        - Distinct values
        - Constant columns
        - High-cardinality columns
        - Numeric-column validity
        - Overall quality summary

    The analyzer does not modify the dataset.
    """

    def __init__(
        self,
        engine: DataEngine,
        table_name: str = "dataset",
    ) -> None:
        self.engine = engine
        self.table_name = table_name

    def build_report(self) -> dict[str, Any]:
        """
        Build the complete data-quality report.
        """

        row_count = self._get_row_count()

        columns = self._analyze_columns(
            row_count=row_count
        )

        duplicate_rows = self._analyze_duplicates(
            row_count=row_count
        )

        summary = self._build_summary(
            row_count=row_count,
            columns=columns,
            duplicate_rows=duplicate_rows,
        )

        return {
            "table_name": self.table_name,
            "row_count": row_count,
            "column_count": len(columns),
            "columns": columns,
            "duplicate_rows": duplicate_rows,
            "summary": summary,
        }

    def _get_row_count(self) -> int:
        result = self.engine.fetch_rows(
            f"SELECT COUNT(*) FROM {self.table_name}"
        )

        return int(result[0][0])

    def _analyze_columns(
        self,
        row_count: int,
    ) -> list[dict[str, Any]]:
        schema = self.engine.inspect_schema(
            self.table_name
        )

        columns: list[dict[str, Any]] = []

        for column in schema:
            column_name = column["column_name"]

            quoted_column = self.engine._quote_identifier(
                column_name
            )

            statistics = self.engine.fetch_dicts(
                f"""
                SELECT
                    COUNT(*) AS row_count,
                    COUNT({quoted_column}) AS non_null_count,
                    COUNT(*) - COUNT({quoted_column}) AS null_count,
                    COUNT(DISTINCT {quoted_column}) AS distinct_count
                FROM {self.table_name}
                """
            )[0]

            null_count = int(
                statistics["null_count"]
            )

            distinct_count = int(
                statistics["distinct_count"]
            )

            if row_count == 0:
                null_percentage = 0.0
                uniqueness_ratio = 0.0
            else:
                null_percentage = (
                    null_count / row_count
                ) * 100.0

                uniqueness_ratio = (
                    distinct_count / row_count
                )

            is_constant = (
                row_count > 0
                and distinct_count <= 1
            )

            is_high_cardinality = (
                row_count > 0
                and uniqueness_ratio >= 0.90
                and distinct_count > 20
            )

            is_numeric = self._is_numeric_type(
                column["column_type"]
            )

            numeric_validity = (
                self._analyze_numeric_validity(
                    quoted_column=quoted_column,
                    is_numeric=is_numeric,
                )
            )

            columns.append(
                {
                    "name": column_name,
                    "sql_type": column["column_type"],
                    "row_count": row_count,
                    "non_null_count": int(
                        statistics["non_null_count"]
                    ),
                    "null_count": null_count,
                    "null_percentage": null_percentage,
                    "distinct_count": distinct_count,
                    "uniqueness_ratio": uniqueness_ratio,
                    "is_constant": is_constant,
                    "is_high_cardinality": (
                        is_high_cardinality
                    ),
                    "is_numeric": is_numeric,
                    "numeric_validity": numeric_validity,
                }
            )

        return columns

    def _analyze_numeric_validity(
        self,
        quoted_column: str,
        is_numeric: bool,
    ) -> dict[str, Any]:
        """
        Inspect numeric columns for basic mathematical validity.

        The current checks identify:
            - Negative infinity
            - Positive infinity
            - NaN

        DuckDB normally represents ordinary numeric values correctly,
        but these checks provide explicit quality signals for numerical
        datasets.
        """

        if not is_numeric:
            return {
                "checked": False,
                "positive_infinity_count": 0,
                "negative_infinity_count": 0,
                "nan_count": 0,
            }

        try:
            result = self.engine.fetch_dicts(
                f"""
                SELECT
                    COUNT(*) FILTER (
                        WHERE isinf({quoted_column})
                        AND {quoted_column} > 0
                    ) AS positive_infinity_count,

                    COUNT(*) FILTER (
                        WHERE isinf({quoted_column})
                        AND {quoted_column} < 0
                    ) AS negative_infinity_count,

                    COUNT(*) FILTER (
                        WHERE isnan({quoted_column})
                    ) AS nan_count
                FROM {self.table_name}
                """
            )[0]

            return {
                "checked": True,
                "positive_infinity_count": int(
                    result["positive_infinity_count"]
                ),
                "negative_infinity_count": int(
                    result["negative_infinity_count"]
                ),
                "nan_count": int(
                    result["nan_count"]
                ),
            }

        except Exception:
            return {
                "checked": False,
                "positive_infinity_count": 0,
                "negative_infinity_count": 0,
                "nan_count": 0,
            }

    def _analyze_duplicates(
        self,
        row_count: int,
    ) -> dict[str, Any]:
        """
        Detect completely duplicated rows.

        A duplicate row is counted when an identical row occurs more
        than once. The first occurrence is considered the original and
        additional occurrences are counted as duplicate rows.
        """

        if row_count == 0:
            return {
                "duplicate_row_count": 0,
                "duplicate_percentage": 0.0,
                "unique_row_count": 0,
            }

        schema = self.engine.inspect_schema(
            self.table_name
        )

        column_expressions = []

        for column in schema:
            column_name = column["column_name"]

            quoted_column = self.engine._quote_identifier(
                column_name
            )

            column_expressions.append(
                f"COALESCE(CAST({quoted_column} AS VARCHAR), "
                f"'<NULL>')"
            )

        row_signature_expression = " || '|' || ".join(
            column_expressions
        )

        duplicate_result = self.engine.fetch_rows(
            f"""
            SELECT
                COUNT(*) - COUNT(
                    DISTINCT md5(
                        {row_signature_expression}
                    )
                )
            FROM {self.table_name}
            """
        )

        duplicate_count = int(
            duplicate_result[0][0]
        )

        unique_row_count = (
            row_count - duplicate_count
        )

        duplicate_percentage = (
            duplicate_count / row_count
        ) * 100.0

        return {
            "duplicate_row_count": duplicate_count,
            "duplicate_percentage": duplicate_percentage,
            "unique_row_count": unique_row_count,
        }
    
    def _build_summary(
        self,
        row_count: int,
        columns: list[dict[str, Any]],
        duplicate_rows: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build a high-level quality summary.

        The summary is descriptive rather than a subjective score.
        """

        columns_with_missing_values = [
            column["name"]
            for column in columns
            if column["null_count"] > 0
        ]

        constant_columns = [
            column["name"]
            for column in columns
            if column["is_constant"]
        ]

        high_cardinality_columns = [
            column["name"]
            for column in columns
            if column["is_high_cardinality"]
        ]

        numeric_problem_columns = [
            column["name"]
            for column in columns
            if (
                column["numeric_validity"][
                    "positive_infinity_count"
                ]
                > 0
                or column["numeric_validity"][
                    "negative_infinity_count"
                ]
                > 0
                or column["numeric_validity"][
                    "nan_count"
                ]
                > 0
            )
        ]

        return {
            "empty_dataset": row_count == 0,
            "columns_with_missing_values": (
                columns_with_missing_values
            ),
            "constant_columns": constant_columns,
            "high_cardinality_columns": (
                high_cardinality_columns
            ),
            "numeric_problem_columns": (
                numeric_problem_columns
            ),
            "has_duplicate_rows": (
                duplicate_rows["duplicate_row_count"] > 0
            ),
            "has_quality_issues": (
                len(columns_with_missing_values) > 0
                or len(constant_columns) > 0
                or len(high_cardinality_columns) > 0
                or len(numeric_problem_columns) > 0
                or duplicate_rows["duplicate_row_count"] > 0
            ),
        }

    @staticmethod
    def _is_numeric_type(
        sql_type: str,
    ) -> bool:
        normalized_type = sql_type.upper()

        numeric_tokens = (
            "TINYINT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
            "HUGEINT",
            "UTINYINT",
            "USMALLINT",
            "UINTEGER",
            "UBIGINT",
            "DECIMAL",
            "NUMERIC",
            "REAL",
            "FLOAT",
            "DOUBLE",
        )

        return any(
            token in normalized_type
            for token in numeric_tokens
        )