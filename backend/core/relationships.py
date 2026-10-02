from __future__ import annotations

from typing import Any

import numpy as np

from backend.core.loader import DataEngine


class RelationshipAnalyzer:
    """
    Deterministic relationship-analysis engine for NEXUS-6.

    Uses NumPy to calculate Pearson and Spearman correlations.
    The engine does not modify the source dataset.
    """

    def __init__(
        self,
        engine: DataEngine,
        table_name: str = "dataset",
    ) -> None:
        self.engine = engine
        self.table_name = table_name

    def build_report(self) -> dict[str, Any]:
        numeric_columns = self._get_numeric_columns()
        relationships: list[dict[str, Any]] = []

        for index, first_column in enumerate(numeric_columns):
            for second_column in numeric_columns[index + 1:]:
                observation = self._analyze_pair(
                    first_column,
                    second_column,
                )

                if observation is not None:
                    relationships.append(observation)

        return {
            "table_name": self.table_name,
            "numeric_column_count": len(numeric_columns),
            "numeric_columns": numeric_columns,
            "relationship_count": len(relationships),
            "relationships": relationships,
        }

    def _get_numeric_columns(self) -> list[str]:
        schema = self.engine.inspect_schema(self.table_name)

        return [
            column["column_name"]
            for column in schema
            if self._is_numeric_type(column["column_type"])
        ]

    def _analyze_pair(
        self,
        first_column: str,
        second_column: str,
    ) -> dict[str, Any] | None:
        first_quoted = self.engine._quote_identifier(first_column)
        second_quoted = self.engine._quote_identifier(second_column)

        rows = self.engine.fetch_rows(
            f"""
            SELECT
                {first_quoted},
                {second_quoted}
            FROM {self.table_name}
            WHERE
                {first_quoted} IS NOT NULL
                AND {second_quoted} IS NOT NULL
            """
        )

        if len(rows) < 2:
            return None

        first_values = np.asarray(
            [row[0] for row in rows],
            dtype=float,
        )
        second_values = np.asarray(
            [row[1] for row in rows],
            dtype=float,
        )

        finite_mask = (
            np.isfinite(first_values)
            & np.isfinite(second_values)
        )

        first_values = first_values[finite_mask]
        second_values = second_values[finite_mask]

        sample_size = len(first_values)

        if sample_size < 2:
            return None

        if (
            np.unique(first_values).size < 2
            or np.unique(second_values).size < 2
        ):
            return {
                "type": "correlation",
                "variables": [
                    first_column,
                    second_column,
                ],
                "sample_size": sample_size,
                "pearson": None,
                "spearman": None,
                "status": "insufficient_variation",
                "source": "relationship_engine",
            }

        pearson_value = float(
            np.corrcoef(first_values, second_values)[0, 1]
        )

        first_ranks = self._rank_values(first_values)
        second_ranks = self._rank_values(second_values)

        spearman_value = float(
            np.corrcoef(first_ranks, second_ranks)[0, 1]
        )

        pearson_value = self._clamp_correlation(pearson_value)
        spearman_value = self._clamp_correlation(spearman_value)

        return {
            "type": "correlation",
            "variables": [
                first_column,
                second_column,
            ],
            "statistic": {
                "pearson": {
                    "value": pearson_value,
                    "p_value": None,
                    "direction": self._get_direction(pearson_value),
                    "strength": self._get_strength(pearson_value),
                },
                "spearman": {
                    "value": spearman_value,
                    "p_value": None,
                    "direction": self._get_direction(spearman_value),
                    "strength": self._get_strength(spearman_value),
                },
            },
            "sample_size": sample_size,
            "method": [
                "pearson",
                "spearman",
            ],
            "significance_testing": "not_calculated",
            "source": "relationship_engine",
        }

    @staticmethod
    def _rank_values(values: np.ndarray) -> np.ndarray:
        """
        Assign average ranks, including for tied values.
        Ranks start at 1.
        """

        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        ranks = np.empty(len(values), dtype=float)

        start = 0

        while start < len(values):
            end = start + 1

            while (
                end < len(values)
                and sorted_values[end] == sorted_values[start]
            ):
                end += 1

            average_rank = (
                (start + 1) + end
            ) / 2.0

            ranks[order[start:end]] = average_rank
            start = end

        return ranks

    @staticmethod
    def _clamp_correlation(value: float) -> float:
        return max(-1.0, min(1.0, value))

    @staticmethod
    def _get_direction(correlation: float) -> str:
        if correlation > 0:
            return "positive"

        if correlation < 0:
            return "negative"

        return "none"

    @staticmethod
    def _get_strength(correlation: float) -> str:
        absolute_value = abs(correlation)

        if absolute_value < 0.30:
            return "weak"

        if absolute_value < 0.70:
            return "moderate"

        return "strong"

    @staticmethod
    def _is_numeric_type(sql_type: str) -> bool:
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