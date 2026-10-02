from __future__ import annotations

from typing import Any

from backend.core.loader import DataEngine


class DatasetIntelligence:
    """
    Builds a deterministic intelligence profile for a dataset.

    The intelligence layer does not use an LLM. It collects factual
    information from DuckDB and organizes it into a structured profile
    that can later be consumed by the NEXUS-6 orchestrator.
    """

    def __init__(
        self,
        engine: DataEngine,
        table_name: str = "dataset",
    ) -> None:
        self.engine = engine
        self.table_name = table_name

    def build_profile(self) -> dict[str, Any]:
        """
        Build the complete dataset intelligence profile.
        """

        metadata = self.engine.inspect_metadata(
            self.table_name
        )

        columns = self._analyze_columns()

        numeric_statistics = (
            self.engine.profile_numeric_columns(
                self.table_name
            )
        )

        capabilities = self._detect_capabilities(
            columns=columns,
            metadata=metadata,
        )

        potential_targets = self._detect_potential_targets(
            columns=columns,
            metadata=metadata,
        )

        return {
            "table_name": self.table_name,
            "row_count": metadata["row_count"],
            "column_count": metadata["column_count"],
            "columns": columns,
            "numeric_statistics": numeric_statistics,
            "potential_targets": potential_targets,
            "capabilities": capabilities,
        }

    def _analyze_columns(self) -> list[dict[str, Any]]:
        """
        Analyze every column in the dataset.

        The analysis includes:
            - SQL type
            - nullability
            - distinct values
            - null count
            - uniqueness ratio
            - semantic category
        """

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

            row_count = statistics["row_count"]
            distinct_count = statistics["distinct_count"]

            if row_count == 0:
                uniqueness_ratio = 0.0
            else:
                uniqueness_ratio = (
                    distinct_count / row_count
                )

            semantic_type = self._infer_semantic_type(
                column_name=column_name,
                sql_type=column["column_type"],
                distinct_count=distinct_count,
                row_count=row_count,
            )

            columns.append(
                {
                    "name": column_name,
                    "sql_type": column["column_type"],
                    "nullable": column["null"] == "YES",
                    "row_count": row_count,
                    "non_null_count": statistics[
                        "non_null_count"
                    ],
                    "null_count": statistics[
                        "null_count"
                    ],
                    "distinct_count": distinct_count,
                    "uniqueness_ratio": uniqueness_ratio,
                    "semantic_type": semantic_type,
                }
            )

        return columns

    @staticmethod
    def _infer_semantic_type(
        column_name: str,
        sql_type: str,
        distinct_count: int,
        row_count: int,
    ) -> str:
        """
        Infer a conservative analytical semantic category.

        Semantic classification considers both the physical SQL type and
        basic characteristics of the column.
        """

        normalized_type = sql_type.upper()
        normalized_name = column_name.lower()

        if any(
            token in normalized_type
            for token in (
                "INT",
                "DECIMAL",
                "DOUBLE",
                "FLOAT",
                "REAL",
                "HUGEINT",
            )
        ):
            return "numeric"

        if "TIMESTAMP" in normalized_type:
            return "datetime"

        if "DATE" in normalized_type:
            return "date"

        if "TIME" in normalized_type:
            return "datetime"

        if "BOOL" in normalized_type:
            return "boolean"

        if not any(
            token in normalized_type
            for token in (
                "CHAR",
                "VARCHAR",
                "TEXT",
                "STRING",
            )
        ):
            return "text"

        if DatasetIntelligence._looks_like_identifier(
            normalized_name
        ):
            return "text"

        if DatasetIntelligence._looks_like_name_column(
            normalized_name
        ):
            return "text"

        if row_count > 0:
            uniqueness_ratio = (
            distinct_count / row_count
            )
            if (
                distinct_count <= 20
                and uniqueness_ratio <= 0.50
                and distinct_count < row_count
            ):
                return "categorical"

        return "text"

    def _detect_potential_targets(
        self,
        columns: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Identify columns that could potentially be prediction targets.

        This is a heuristic, not a machine-learning conclusion.

        Numeric, boolean, and low-cardinality categorical columns are
        considered possible targets. Identifier-like and name-like
        columns are excluded.
        """

        potential_targets: list[dict[str, Any]] = []

        for column in columns:
            name = column["name"]
            normalized_name = name.lower()

            if self._looks_like_identifier(
                normalized_name
            ):
                continue

            if self._looks_like_name_column(
                normalized_name
            ):
                continue

            if column["semantic_type"] not in (
                "numeric",
                "categorical",
                "boolean",
            ):
                continue

            if column["null_count"] == metadata["row_count"]:
                continue

            potential_targets.append(
                {
                    "column": name,
                    "semantic_type": column[
                        "semantic_type"
                    ],
                    "distinct_count": column[
                        "distinct_count"
                    ],
                    "reason": (
                        "Column has a structured analytical type "
                        "and may be suitable as a supervised "
                        "learning target."
                    ),
                }
            )

        return potential_targets

    @staticmethod
    def _looks_like_identifier(
        column_name: str,
    ) -> bool:
        """
        Detect common identifier-style column names.
        """

        identifier_names = {
            "id",
            "uuid",
            "guid",
            "identifier",
        }

        identifier_suffixes = (
            "_id",
            "_uuid",
            "_guid",
            "_identifier",
        )

        if column_name in identifier_names:
            return True

        return column_name.endswith(
            identifier_suffixes
        )

    @staticmethod
    def _looks_like_name_column(
        column_name: str,
    ) -> bool:
        """
        Detect columns that appear to contain personal or entity names.
        """

        name_tokens = (
            "name",
            "full_name",
            "first_name",
            "last_name",
            "employee_name",
            "customer_name",
            "person_name",
            "student_name",
            "patient_name",
            "company_name",
            "organization_name",
        )

        if column_name in name_tokens:
            return True

        return column_name.endswith(
            (
                "_name",
            )
        )

    @staticmethod
    def _detect_capabilities(
        columns: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> dict[str, bool]:
        """
        Determine which broad analytical operations the dataset can
        potentially support.

        These are capability signals, not guarantees that an analysis
        will succeed.
        """

        numeric_count = sum(
            column["semantic_type"] == "numeric"
            for column in columns
        )

        categorical_count = sum(
            column["semantic_type"] == "categorical"
            for column in columns
        )

        date_count = sum(
            column["semantic_type"] == "date"
            or column["semantic_type"] == "datetime"
            for column in columns
        )

        return {
            "query": metadata["column_count"] > 0,
            "overview": metadata["row_count"] > 0,
            "audit": metadata["column_count"] > 0,
            "repair": metadata["row_count"] > 0,
            "relationships": numeric_count >= 2,
            "categorical_analysis": categorical_count >= 1,
            "time_series_analysis": date_count >= 1,
            "predictive_modeling": (
                numeric_count >= 1
                and metadata["row_count"] >= 10
            ),
            "engineering_analysis": (
                metadata["column_count"] > 0
            ),
        }