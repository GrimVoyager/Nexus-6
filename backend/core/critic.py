from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from backend.core.evidence import DatasetEvidencePackage
from backend.core.executor import TierExecutionResult


class CriticStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    CONTRADICTORY = "CONTRADICTORY"


class CriticDecision(BaseModel):
    """
    Structured decision produced by the NEXUS-6 critic.

    The critic evaluates evidence sufficiency. It does not generate
    the final analytical answer and does not modify the dataset.
    """

    status: CriticStatus
    replan_required: bool
    reasons: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    executed_tiers: list[int] = Field(default_factory=list)
    partial_tiers: list[int] = Field(default_factory=list)
    skipped_tiers: list[int] = Field(default_factory=list)
    missing_requirements: list[str] = Field(
        default_factory=list
    )
    contradiction_count: int = 0


class CriticValidator:
    """
    Deterministic evidence critic for NEXUS-6.

    The critic does not:
        - use an LLM
        - modify source data
        - modify evidence
        - create new observations
        - reinterpret numerical statistics

    It only evaluates whether the available execution evidence
    is adequate for the supplied objective.
    """

    def evaluate(
        self,
        *,
        objective: str,
        evidence: DatasetEvidencePackage,
        execution_result: TierExecutionResult,
    ) -> CriticDecision:
        normalized_objective = objective.strip().lower()

        evidence_count = len(
            evidence.observations
        )

        executed_tiers = list(
            execution_result.executed_tiers
        )

        partial_tiers = list(
            execution_result.partial_tiers
        )

        skipped_tiers = list(
            execution_result.skipped_tiers
        )

        reasons: list[str] = []
        missing_requirements: list[str] = []

        contradiction_count = (
            self._count_contradictions(
                evidence
            )
        )

        if contradiction_count > 0:
            reasons.append(
                "Conflicting observations were detected "
                "for the same variables and statistic."
            )

            return CriticDecision(
                status=CriticStatus.CONTRADICTORY,
                replan_required=True,
                reasons=reasons,
                evidence_count=evidence_count,
                executed_tiers=executed_tiers,
                partial_tiers=partial_tiers,
                skipped_tiers=skipped_tiers,
                missing_requirements=missing_requirements,
                contradiction_count=contradiction_count,
            )

        if evidence_count == 0:
            reasons.append(
                "No observations are available for evaluation."
            )

            return CriticDecision(
                status=CriticStatus.INSUFFICIENT,
                replan_required=True,
                reasons=reasons,
                evidence_count=evidence_count,
                executed_tiers=executed_tiers,
                partial_tiers=partial_tiers,
                skipped_tiers=skipped_tiers,
                missing_requirements=[
                    "at_least_one_observation"
                ],
                contradiction_count=0,
            )

        if self._requires_query(
            normalized_objective
        ):
            if 1 not in executed_tiers:
                missing_requirements.append(
                    "query_execution"
                )

        if self._requires_repair(
            normalized_objective
        ):
            if 4 not in executed_tiers and 4 not in partial_tiers:
                missing_requirements.append(
                    "repair_analysis"
                )

        if self._requires_intelligence(
            normalized_objective
        ):
            if 5 not in executed_tiers:
                missing_requirements.append(
                    "intelligence_analysis"
                )

        if self._requires_engineering(
            normalized_objective
        ):
            if 6 not in executed_tiers and 6 not in partial_tiers:
                missing_requirements.append(
                    "engineering_analysis"
                )

        if partial_tiers:
            reasons.append(
                "One or more selected tiers were only partially "
                "executed."
            )

        if missing_requirements:
            reasons.append(
                "The available evidence does not satisfy all "
                "requirements for the requested objective."
            )

            return CriticDecision(
                status=CriticStatus.INSUFFICIENT,
                replan_required=True,
                reasons=reasons,
                evidence_count=evidence_count,
                executed_tiers=executed_tiers,
                partial_tiers=partial_tiers,
                skipped_tiers=skipped_tiers,
                missing_requirements=missing_requirements,
                contradiction_count=0,
            )

        reasons.append(
            "Available evidence satisfies the deterministic "
            "requirements identified for the objective."
        )

        return CriticDecision(
            status=CriticStatus.SUFFICIENT,
            replan_required=False,
            reasons=reasons,
            evidence_count=evidence_count,
            executed_tiers=executed_tiers,
            partial_tiers=partial_tiers,
            skipped_tiers=skipped_tiers,
            missing_requirements=[],
            contradiction_count=0,
        )

    @staticmethod
    def _requires_query(
        objective: str,
    ) -> bool:
        query_terms = (
            "query",
            "sql",
            "select",
            "filter",
            "find rows",
            "retrieve",
        )

        return any(
            term in objective
            for term in query_terms
        )

    @staticmethod
    def _requires_repair(
        objective: str,
    ) -> bool:
        repair_terms = (
            "repair",
            "clean",
            "fix",
            "duplicate",
            "missing value",
            "missing values",
            "data quality",
        )

        return any(
            term in objective
            for term in repair_terms
        )

    @staticmethod
    def _requires_intelligence(
        objective: str,
    ) -> bool:
        intelligence_terms = (
            "relationship",
            "correlation",
            "predict",
            "prediction",
            "model",
            "forecast",
            "anomaly",
            "outlier",
            "classification",
            "regression",
        )

        return any(
            term in objective
            for term in intelligence_terms
        )

    @staticmethod
    def _requires_engineering(
        objective: str,
    ) -> bool:
        engineering_terms = (
            "migration",
            "migrate",
            "ddl",
            "dml",
            "dbt",
            "pyspark",
            "transform",
            "etl",
        )

        return any(
            term in objective
            for term in engineering_terms
        )

    @staticmethod
    def _count_contradictions(
        evidence: DatasetEvidencePackage,
    ) -> int:
        """
        Detect direct contradictions where observations describe
        the same variables, statistic name, method, and sample size
        but contain different statistic values.

        This intentionally uses a conservative definition of
        contradiction. Different methods or sample sizes are not
        automatically considered contradictory.
        """

        def make_hashable(value: Any) -> Any:
            if isinstance(value, dict):
                return tuple(
                    sorted(
                        (
                            str(key),
                            make_hashable(item),
                        )
                        for key, item in value.items()
                    )
                )

            if isinstance(value, (list, tuple)):
                return tuple(
                    make_hashable(item)
                    for item in value
                )

            if isinstance(value, set):
                return tuple(
                    sorted(
                        make_hashable(item)
                        for item in value
                    )
                )

            return value

        groups: dict[
            tuple[Any, ...],
            list[Any]
        ] = {}

        for observation in evidence.observations:
            statistic = observation.statistic

            if not isinstance(
                statistic,
                dict,
            ):
                continue

            statistic_items = make_hashable(
                statistic
            )

            key = (
                tuple(observation.variables),
                make_hashable(observation.method),
                observation.sample_size,
            )

            groups.setdefault(
                key,
                [],
            ).append(
                statistic_items
            )

        contradictions = 0

        for values in groups.values():
            unique_values = set(values)

            if len(unique_values) > 1:
                contradictions += 1

        return contradictions

    @staticmethod
    def validate_decision(
        decision: CriticDecision,
    ) -> None:
        if decision.status == CriticStatus.SUFFICIENT:
            if decision.replan_required:
                raise ValueError(
                    "A SUFFICIENT critic decision cannot "
                    "require replanning."
                )

        if decision.status in {
            CriticStatus.INSUFFICIENT,
            CriticStatus.CONTRADICTORY,
        }:
            if not decision.replan_required:
                raise ValueError(
                    "An INSUFFICIENT or CONTRADICTORY decision "
                    "must require replanning."
                )

        if decision.evidence_count < 0:
            raise ValueError(
                "Evidence count cannot be negative."
            )

        if decision.contradiction_count < 0:
            raise ValueError(
                "Contradiction count cannot be negative."
            )