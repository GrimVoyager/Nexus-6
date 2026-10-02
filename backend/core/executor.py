from __future__ import annotations

from typing import Any

from backend.core.evidence import DatasetEvidencePackage
from backend.core.intelligence import DatasetIntelligence
from backend.core.observation import ObservationStore
from backend.core.observation_collector import ObservationCollector
from backend.core.planner import (
    AnalysisTier,
    ExecutionPlan,
    TierStatus,
)
from backend.core.quality import DataQualityAnalyzer
from backend.core.relationships import RelationshipAnalyzer
from backend.core.statistics import StatisticalAnalyzer


class TierExecutionResult:
    """
    Result produced by the NEXUS-6 Tier Executor.

    The result distinguishes between:
        - What the orchestrator planned.
        - What was actually executed.
        - What remains unsupported.
        - What evidence was produced by execution.
    """

    def __init__(
        self,
        plan: ExecutionPlan,
        executed_tiers: list[int],
        partial_tiers: list[int],
        skipped_tiers: list[int],
        observations: ObservationStore,
        execution_details: dict[str, Any],
    ) -> None:
        self.plan = plan
        self.executed_tiers = executed_tiers
        self.partial_tiers = partial_tiers
        self.skipped_tiers = skipped_tiers
        self.observations = observations
        self.execution_details = execution_details

    @property
    def observation_count(self) -> int:
        return self.observations.count()

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed_tiers": self.executed_tiers,
            "partial_tiers": self.partial_tiers,
            "skipped_tiers": self.skipped_tiers,
            "observation_count": self.observation_count,
            "execution_details": self.execution_details,
            "observations": self.observations.to_dict(),
        }


class TierExecutor:
    """
    Deterministic execution layer for NEXUS-6.

    The executor consumes an already-created ExecutionPlan and
    performs only the work authorized by that plan.

    It does not:
        - detect the domain
        - reinterpret the objective
        - change tier decisions
        - silently mutate source data
        - use an LLM
    """

    def __init__(
        self,
        *,
        intelligence: DatasetIntelligence,
        quality_analyzer: DataQualityAnalyzer,
        relationship_analyzer: RelationshipAnalyzer,
        statistical_analyzer: StatisticalAnalyzer,
        observation_collector: ObservationCollector,
    ) -> None:
        self.intelligence = intelligence
        self.quality_analyzer = quality_analyzer
        self.relationship_analyzer = relationship_analyzer
        self.statistical_analyzer = statistical_analyzer
        self.observation_collector = observation_collector

    def execute(
        self,
        *,
        evidence: DatasetEvidencePackage,
        plan: ExecutionPlan,
    ) -> TierExecutionResult:
        """
        Execute the supplied plan.

        Existing evidence is preserved. New execution evidence is
        collected separately and returned as an ObservationStore.
        """

        executed_tiers: list[int] = []
        partial_tiers: list[int] = []
        skipped_tiers: list[int] = []

        observations = ObservationStore()

        execution_details: dict[str, Any] = {}

        for decision in plan.tier_decisions:
            tier_number = int(decision.tier)

            if decision.status == TierStatus.EXECUTE.value:
                result = self._execute_tier(
                    tier=decision.tier,
                    evidence=evidence,
                    plan=plan,
                )

                executed_tiers.append(tier_number)

                execution_details[str(tier_number)] = result

                self._collect_result_observations(
                    result=result,
                    target_store=observations,
                )

            elif decision.status == TierStatus.PARTIAL.value:
                result = self._execute_partial_tier(
                    tier=decision.tier,
                    evidence=evidence,
                    plan=plan,
                )

                partial_tiers.append(tier_number)

                execution_details[str(tier_number)] = result

                self._collect_result_observations(
                    result=result,
                    target_store=observations,
                )

            elif decision.status == TierStatus.NOT_APPLICABLE.value:
                skipped_tiers.append(tier_number)

                execution_details[str(tier_number)] = {
                    "status": "NOT_EXECUTED",
                    "reason": decision.reason,
                }

            else:
                raise ValueError(
                    "Unsupported tier status: "
                    f"{decision.status}"
                )

        return TierExecutionResult(
            plan=plan,
            executed_tiers=executed_tiers,
            partial_tiers=partial_tiers,
            skipped_tiers=skipped_tiers,
            observations=observations,
            execution_details=execution_details,
        )

    def _execute_tier(
        self,
        *,
        tier: AnalysisTier,
        evidence: DatasetEvidencePackage,
        plan: ExecutionPlan,
    ) -> dict[str, Any]:
        if tier == AnalysisTier.QUERY:
            return self._execute_query_tier(
                plan=plan
            )

        if tier == AnalysisTier.OVERVIEW:
            return self._execute_overview_tier()

        if tier == AnalysisTier.AUDIT:
            return self._execute_audit_tier()

        if tier == AnalysisTier.REPAIR:
            return self._execute_repair_tier(
                evidence=evidence
            )

        if tier == AnalysisTier.INTELLIGENCE:
            return self._execute_intelligence_tier()

        if tier == AnalysisTier.ENGINEERING:
            return self._execute_engineering_tier()

        raise ValueError(
            f"Unsupported analysis tier: {tier}"
        )

    def _execute_partial_tier(
        self,
        *,
        tier: AnalysisTier,
        evidence: DatasetEvidencePackage,
        plan: ExecutionPlan,
    ) -> dict[str, Any]:
        if tier == AnalysisTier.REPAIR:
            return self._execute_repair_tier(
                evidence=evidence
            )

        if tier == AnalysisTier.QUERY:
            return {
                "status": "PARTIAL",
                "executed": False,
                "reason": (
                    "Query execution requires a validated "
                    "query compiler before source execution."
                ),
            }

        if tier == AnalysisTier.ENGINEERING:
            return {
                "status": "PARTIAL",
                "executed": False,
                "reason": (
                    "Engineering execution requires a dedicated "
                    "migration or transformation engine."
                ),
            }

        if tier == AnalysisTier.INTELLIGENCE:
            return {
                "status": "PARTIAL",
                "executed": False,
                "reason": (
                    "Insufficient capability exists for full "
                    "intelligence execution."
                ),
            }

        return {
            "status": "PARTIAL",
            "executed": False,
            "reason": (
                "This tier does not currently have a partial "
                "execution path."
            ),
        }

    def _execute_query_tier(
        self,
        *,
        plan: ExecutionPlan,
    ) -> dict[str, Any]:
        if not plan.primary_sql:
            return {
                "status": "NOT_EXECUTED",
                "executed": False,
                "reason": (
                    "Query tier was selected, but no validated "
                    "primary SQL statement is available."
                ),
            }

        return {
            "status": "PLANNED",
            "executed": False,
            "reason": (
                "Query execution is reserved for the validated "
                "SQL compiler and query execution engine."
            ),
            "sql": plan.primary_sql,
        }

    def _execute_overview_tier(
        self,
    ) -> dict[str, Any]:
        profile = self.intelligence.build_profile()

        return {
            "status": "EXECUTED",
            "executed": True,
            "engine": "dataset_intelligence",
            "result": profile,
        }

    def _execute_audit_tier(
        self,
    ) -> dict[str, Any]:
        quality_report = (
            self.quality_analyzer.build_report()
        )

        statistical_report = (
            self.statistical_analyzer.build_report()
        )

        return {
            "status": "EXECUTED",
            "executed": True,
            "engine": "quality_and_statistics",
            "quality_report": quality_report,
            "statistical_report": statistical_report,
        }

    def _execute_repair_tier(
        self,
        *,
        evidence: DatasetEvidencePackage,
    ) -> dict[str, Any]:
        repair_candidates = [
            {
                "observation_id": observation.observation_id,
                "type": observation.type,
                "variables": observation.variables,
                "statistic": observation.statistic,
            }
            for observation in evidence.observations
            if observation.type in {
                "missing_values",
                "duplicate_rows",
                "numeric_validity",
                "constant_column",
                "high_cardinality",
                "outliers",
            }
        ]

        return {
            "status": "PLANNED",
            "executed": False,
            "mutation_performed": False,
            "reason": (
                "Repair analysis identifies candidates but does not "
                "modify the source dataset."
            ),
            "repair_candidates": repair_candidates,
        }

    def _execute_intelligence_tier(
        self,
    ) -> dict[str, Any]:
        relationship_report = (
            self.relationship_analyzer.build_report()
        )

        statistical_report = (
            self.statistical_analyzer.build_report()
        )

        return {
            "status": "EXECUTED",
            "executed": True,
            "engine": "relationships_and_statistics",
            "relationship_report": relationship_report,
            "statistical_report": statistical_report,
        }

    def _execute_engineering_tier(
        self,
    ) -> dict[str, Any]:
        return {
            "status": "PLANNED",
            "executed": False,
            "mutation_performed": False,
            "reason": (
                "Engineering execution requires a dedicated "
                "engineering engine."
            ),
        }

    @staticmethod
    def _collect_result_observations(
        *,
        result: dict[str, Any],
        target_store: ObservationStore,
    ) -> None:
        """
        Preserve the current deterministic observation model.

        Analytical engines already know how to create standardized
        observations. This method will connect future executor
        adapters to the ObservationStore.
        """

        result_store = result.get(
            "observation_store"
        )

        if not isinstance(
            result_store,
            ObservationStore,
        ):
            return

        target_store.add_many(
            result_store.all()
        )