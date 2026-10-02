from __future__ import annotations

from typing import Any

from backend.core.observation import (
    Observation,
    ObservationStore,
    create_observation_id,
)
from backend.core.quality import DataQualityAnalyzer
from backend.core.relationships import RelationshipAnalyzer
from backend.core.statistics import StatisticalAnalyzer


class ObservationCollector:
    """
    Converts deterministic analytical engine reports into standardized
    NEXUS-6 observations.

    The collector does not perform new statistical calculations.

    Its responsibility is to:
        1. Run the existing analytical engines.
        2. Convert their results into Observation objects.
        3. Store those observations in one ObservationStore.
        4. Return the unified evidence package.

    This creates a clean boundary between deterministic analysis and
    future reasoning/orchestration layers.
    """

    def __init__(
        self,
        quality_analyzer: DataQualityAnalyzer,
        relationship_analyzer: RelationshipAnalyzer,
        statistical_analyzer: StatisticalAnalyzer,
    ) -> None:
        self.quality_analyzer = quality_analyzer
        self.relationship_analyzer = relationship_analyzer
        self.statistical_analyzer = statistical_analyzer

    def collect(self) -> ObservationStore:
        """
        Run all configured analytical engines and collect their
        results into one ObservationStore.
        """

        store = ObservationStore()

        quality_report = (
            self.quality_analyzer.build_report()
        )

        relationship_report = (
            self.relationship_analyzer.build_report()
        )

        statistical_report = (
            self.statistical_analyzer.build_report()
        )

        self._collect_quality_observations(
            report=quality_report,
            store=store,
        )

        self._collect_relationship_observations(
            report=relationship_report,
            store=store,
        )

        self._collect_statistical_observations(
            report=statistical_report,
            store=store,
        )

        return store

    def _collect_quality_observations(
        self,
        report: dict[str, Any],
        store: ObservationStore,
    ) -> None:
        """
        Convert data-quality report information into observations.
        """

        for column in report["columns"]:
            column_name = column["name"]

            if column["null_count"] > 0:
                observation = self._create_observation(
                    observation_type="missing_values",
                    variables=[column_name],
                    statistic={
                        "null_count": column["null_count"],
                        "null_percentage": column[
                            "null_percentage"
                        ],
                    },
                    sample_size=report["row_count"],
                    method="null_analysis",
                    source="quality_engine",
                    metadata={
                        "sql_type": column["sql_type"],
                        "distinct_count": column[
                            "distinct_count"
                        ],
                    },
                )

                store.add(observation)

            if column["is_constant"]:
                observation = self._create_observation(
                    observation_type="constant_column",
                    variables=[column_name],
                    statistic={
                        "distinct_count": column[
                            "distinct_count"
                        ],
                    },
                    sample_size=report["row_count"],
                    method="distinct_value_analysis",
                    source="quality_engine",
                    metadata={
                        "sql_type": column["sql_type"],
                    },
                )

                store.add(observation)

            if column["is_high_cardinality"]:
                observation = self._create_observation(
                    observation_type="high_cardinality",
                    variables=[column_name],
                    statistic={
                        "distinct_count": column[
                            "distinct_count"
                        ],
                        "uniqueness_ratio": column[
                            "uniqueness_ratio"
                        ],
                    },
                    sample_size=report["row_count"],
                    method="cardinality_analysis",
                    source="quality_engine",
                    metadata={
                        "sql_type": column["sql_type"],
                    },
                )

                store.add(observation)

            numeric_validity = column[
                "numeric_validity"
            ]

            if not numeric_validity["checked"]:
                continue

            if (
                numeric_validity[
                    "positive_infinity_count"
                ] > 0
                or numeric_validity[
                    "negative_infinity_count"
                ] > 0
                or numeric_validity[
                    "nan_count"
                ] > 0
            ):
                observation = self._create_observation(
                    observation_type="numeric_validity",
                    variables=[column_name],
                    statistic={
                        "positive_infinity_count": (
                            numeric_validity[
                                "positive_infinity_count"
                            ]
                        ),
                        "negative_infinity_count": (
                            numeric_validity[
                                "negative_infinity_count"
                            ]
                        ),
                        "nan_count": numeric_validity[
                            "nan_count"
                        ],
                    },
                    sample_size=report["row_count"],
                    method="numeric_validity_analysis",
                    source="quality_engine",
                    metadata={
                        "sql_type": column["sql_type"],
                    },
                )

                store.add(observation)

        duplicate_rows = report["duplicate_rows"]

        if duplicate_rows["duplicate_row_count"] > 0:
            observation = self._create_observation(
                observation_type="duplicate_rows",
                variables=[],
                statistic={
                    "duplicate_row_count": duplicate_rows[
                        "duplicate_row_count"
                    ],
                    "duplicate_percentage": duplicate_rows[
                        "duplicate_percentage"
                    ],
                    "unique_row_count": duplicate_rows[
                        "unique_row_count"
                    ],
                },
                sample_size=report["row_count"],
                method="duplicate_row_analysis",
                source="quality_engine",
            )

            store.add(observation)

    def _collect_relationship_observations(
        self,
        report: dict[str, Any],
        store: ObservationStore,
    ) -> None:
        """
        Convert relationship-analysis results into observations.
        """

        for relationship in report["relationships"]:
            variables = relationship["variables"]

            if relationship.get("status") == (
                "insufficient_variation"
            ):
                observation = self._create_observation(
                    observation_type="correlation",
                    variables=variables,
                    statistic={
                        "status": "insufficient_variation",
                        "pearson": None,
                        "spearman": None,
                    },
                    sample_size=relationship[
                        "sample_size"
                    ],
                    method="correlation",
                    source="relationship_engine",
                )

                store.add(observation)

                continue

            statistic = relationship["statistic"]

            observation = self._create_observation(
                observation_type="correlation",
                variables=variables,
                statistic={
                    "pearson": statistic["pearson"],
                    "spearman": statistic["spearman"],
                },
                sample_size=relationship[
                    "sample_size"
                ],
                method=relationship["method"],
                source="relationship_engine",
                metadata={
                    "significance_testing": relationship[
                        "significance_testing"
                    ],
                },
            )

            store.add(observation)

    def _collect_statistical_observations(
        self,
        report: dict[str, Any],
        store: ObservationStore,
    ) -> None:
        """
        Convert descriptive statistical results into observations.
        """

        for statistic in report["statistics"]:
            column_name = statistic["column"]

            if statistic["status"] != "success":
                observation = self._create_observation(
                    observation_type="distribution",
                    variables=[column_name],
                    statistic={
                        "status": statistic["status"],
                    },
                    sample_size=statistic[
                        "sample_size"
                    ],
                    method="descriptive_statistics",
                    source="statistics_engine",
                )

                store.add(observation)

                continue

            observation = self._create_observation(
                observation_type="distribution",
                variables=[column_name],
                statistic={
                    "minimum": statistic["minimum"],
                    "maximum": statistic["maximum"],
                    "mean": statistic["mean"],
                    "median": statistic["median"],
                    "standard_deviation": statistic[
                        "standard_deviation"
                    ],
                    "variance": statistic["variance"],
                    "quartiles": statistic[
                        "quartiles"
                    ],
                    "percentiles": statistic[
                        "percentiles"
                    ],
                    "zero_count": statistic[
                        "zero_count"
                    ],
                    "negative_count": statistic[
                        "negative_count"
                    ],
                },
                sample_size=statistic[
                    "sample_size"
                ],
                method="descriptive_statistics",
                source="statistics_engine",
            )

            store.add(observation)

            outlier_observation = self._create_observation(
                observation_type="outliers",
                variables=[column_name],
                statistic={
                    "outlier_count": statistic[
                        "outlier_count"
                    ],
                    "outlier_percentage": statistic[
                        "outlier_percentage"
                    ],
                    "lower_bound": statistic[
                        "outlier_bounds"
                    ]["lower"],
                    "upper_bound": statistic[
                        "outlier_bounds"
                    ]["upper"],
                },
                sample_size=statistic[
                    "sample_size"
                ],
                method="iqr_1_5_rule",
                source="statistics_engine",
            )

            store.add(outlier_observation)

            skewness_observation = self._create_observation(
                observation_type="skewness",
                variables=[column_name],
                statistic={
                    "skewness": statistic[
                        "skewness"
                    ],
                },
                sample_size=statistic[
                    "sample_size"
                ],
                method="adjusted_fisher_pearson",
                source="statistics_engine",
            )

            store.add(skewness_observation)

    @staticmethod
    def _create_observation(
        *,
        observation_type: str,
        variables: list[str],
        statistic: dict[str, Any],
        sample_size: int | None,
        method: str | list[str] | None,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> Observation:
        observation_id = create_observation_id(
            observation_type=observation_type,
            variables=variables,
            source=source,
            method=method,
        )

        return Observation(
            observation_id=observation_id,
            type=observation_type,
            variables=variables,
            statistic=statistic,
            sample_size=sample_size,
            method=method,
            source=source,
            metadata=metadata or {},
        )