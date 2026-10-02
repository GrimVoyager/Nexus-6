from __future__ import annotations

from typing import Any

import numpy as np

from backend.core.loader import DataEngine


class StatisticalAnalyzer:
    """
    Deterministic statistical profiling engine for NEXUS-6.

    The analyzer calculates descriptive statistics for numeric columns
    using DuckDB for data access and NumPy for numerical operations.

    The analyzer does not modify the source dataset.
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
        Build the complete statistical profile.
        """

        numeric_columns = self._get_numeric_columns()

        statistics: list[dict[str, Any]] = []

        for column_name in numeric_columns:
            result = self._analyze_column(
                column_name
            )

            statistics.append(result)

        return {
            "table_name": self.table_name,
            "numeric_column_count": len(
                numeric_columns
            ),
            "numeric_columns": numeric_columns,
            "statistics": statistics,
        }

    def _get_numeric_columns(self) -> list[str]:
        schema = self.engine.inspect_schema(
            self.table_name
        )

        numeric_columns: list[str] = []

        for column in schema:
            if self._is_numeric_type(
                column["column_type"]
            ):
                numeric_columns.append(
                    column["column_name"]
                )

        return numeric_columns

    def _analyze_column(
        self,
        column_name: str,
    ) -> dict[str, Any]:
        quoted_column = self.engine._quote_identifier(
            column_name
        )

        rows = self.engine.fetch_rows(
            f"""
            SELECT {quoted_column}
            FROM {self.table_name}
            WHERE {quoted_column} IS NOT NULL
            """
        )

        values = np.asarray(
            [row[0] for row in rows],
            dtype=float,
        )

        finite_values = values[
            np.isfinite(values)
        ]

        sample_size = len(finite_values)

        if sample_size == 0:
            return {
                "column": column_name,
                "sample_size": 0,
                "minimum": None,
                "maximum": None,
                "mean": None,
                "median": None,
                "standard_deviation": None,
                "variance": None,
                "quartiles": {
                    "q1": None,
                    "q2": None,
                    "q3": None,
                },
                "percentiles": {
                    "p5": None,
                    "p25": None,
                    "p50": None,
                    "p75": None,
                    "p95": None,
                },
                "zero_count": 0,
                "negative_count": 0,
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "skewness": None,
                "status": "no_valid_numeric_values",
                "source": "statistics_engine",
            }

        minimum = float(
            np.min(finite_values)
        )

        maximum = float(
            np.max(finite_values)
        )

        mean = float(
            np.mean(finite_values)
        )

        median = float(
            np.median(finite_values)
        )

        if sample_size >= 2:
            standard_deviation = float(
                np.std(
                    finite_values,
                    ddof=1,
                )
            )

            variance = float(
                np.var(
                    finite_values,
                    ddof=1,
                )
            )
        else:
            standard_deviation = None
            variance = None

        q1, q2, q3 = np.percentile(
            finite_values,
            [25, 50, 75],
        )

        p5, p25, p50, p75, p95 = np.percentile(
            finite_values,
            [5, 25, 50, 75, 95],
        )

        zero_count = int(
            np.count_nonzero(
                finite_values == 0
            )
        )

        negative_count = int(
            np.count_nonzero(
                finite_values < 0
            )
        )

        iqr = float(q3 - q1)

        lower_bound = float(
            q1 - (1.5 * iqr)
        )

        upper_bound = float(
            q3 + (1.5 * iqr)
        )

        outlier_mask = (
            (finite_values < lower_bound)
            | (finite_values > upper_bound)
        )

        outlier_count = int(
            np.count_nonzero(outlier_mask)
        )

        outlier_percentage = (
            outlier_count / sample_size
        ) * 100.0

        skewness = self._calculate_skewness(
            finite_values,
            mean,
            standard_deviation,
        )

        return {
            "column": column_name,
            "sample_size": sample_size,
            "minimum": minimum,
            "maximum": maximum,
            "mean": mean,
            "median": median,
            "standard_deviation": standard_deviation,
            "variance": variance,
            "quartiles": {
                "q1": float(q1),
                "q2": float(q2),
                "q3": float(q3),
                "iqr": iqr,
            },
            "percentiles": {
                "p5": float(p5),
                "p25": float(p25),
                "p50": float(p50),
                "p75": float(p75),
                "p95": float(p95),
            },
            "zero_count": zero_count,
            "negative_count": negative_count,
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_percentage,
            "outlier_bounds": {
                "lower": lower_bound,
                "upper": upper_bound,
            },
            "skewness": skewness,
            "status": "success",
            "source": "statistics_engine",
        }

    @staticmethod
    def _calculate_skewness(
        values: np.ndarray,
        mean: float,
        standard_deviation: float | None,
    ) -> float | None:
        """
        Calculate the adjusted Fisher-Pearson sample skewness.

        A value near zero indicates approximate symmetry.
        Positive values indicate a longer right tail.
        Negative values indicate a longer left tail.

        The adjusted estimator requires at least three observations
        and non-zero sample standard deviation.
        """

        sample_size = len(values)

        if (
            sample_size < 3
            or standard_deviation is None
            or standard_deviation == 0
        ):
            return None

        centered = values - mean

        m2 = np.mean(
            centered ** 2
        )

        m3 = np.mean(
            centered ** 3
        )

        if m2 == 0:
            return None

        population_skewness = (
            m3 / (m2 ** 1.5)
        )

        adjustment = (
            np.sqrt(sample_size * (sample_size - 1))
            / (sample_size - 2)
        )

        return float(
            adjustment * population_skewness
        )

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