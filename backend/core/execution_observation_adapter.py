from __future__ import annotations

from backend.core.executor import TierExecutionResult
from backend.core.observation import (
    Observation,
    ObservationStore,
    create_observation_id,
)


class ExecutionObservationAdapter:
    """
    Converts deterministic tier execution results into
    standardized Observation objects.

    The adapter records what execution actually established.

    It does not:
        - invent analytical findings
        - interpret numerical results
        - modify the source dataset
        - execute another analysis
        - use an LLM

    Execution observations are provenance/evidence observations.
    Analytical observations remain the responsibility of the
    underlying analytical engines.
    """

    def collect(
        self,
        execution_result: TierExecutionResult,
    ) -> ObservationStore:
        store = ObservationStore()

        for tier in execution_result.executed_tiers:
            observation = self._create_tier_observation(
                execution_result=execution_result,
                tier=tier,
                status="EXECUTED",
            )
            store.add(observation)

        for tier in execution_result.partial_tiers:
            observation = self._create_tier_observation(
                execution_result=execution_result,
                tier=tier,
                status="PARTIAL",
            )
            store.add(observation)

        return store

    @staticmethod
    def _create_tier_observation(
        *,
        execution_result: TierExecutionResult,
        tier: int,
        status: str,
    ) -> Observation:
        details = (
            execution_result.execution_details
            or {}
        )

        tier_details = (
            details.get(str(tier))
            or details.get(tier)
            or {}
        )

        if not isinstance(tier_details, dict):
            tier_details = {
                "detail": str(tier_details)
            }

        statistic = {
            "execution_status": status,
            "tier": tier,
            "details": tier_details,
        }

        observation_id = create_observation_id(
            observation_type="tier_execution",
            variables=[
                f"tier_{tier}"
            ],
            source="execution_observation_adapter",
            method="deterministic_execution_adapter",
        )

        return Observation(
            observation_id=observation_id,
            type="tier_execution",
            variables=[
                f"tier_{tier}"
            ],
            statistic=statistic,
            sample_size=1,
            method="deterministic_execution_adapter",
            source="execution_observation_adapter",
            metadata={
                "execution_status": status,
                "tier": tier,
                "executed_tiers": list(
                    execution_result.executed_tiers
                ),
                "partial_tiers": list(
                    execution_result.partial_tiers
                ),
                "skipped_tiers": list(
                    execution_result.skipped_tiers
                ),
            },
        )