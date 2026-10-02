from __future__ import annotations

from backend.core.critic import (
    CriticStatus,
    CriticValidator,
)
from backend.core.evidence import (
    DatasetEvidencePackage,
    EvidenceSummary,
)
from backend.core.executor import TierExecutionResult
from backend.core.observation import (
    Observation,
    ObservationStore,
)
from backend.core.planner import ExecutionPlan


def create_evidence(
    observations: list[Observation],
) -> DatasetEvidencePackage:
    source_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}

    for observation in observations:
        source_counts[observation.source] = (
            source_counts.get(
                observation.source,
                0,
            )
            + 1
        )

        type_counts[observation.type] = (
            type_counts.get(
                observation.type,
                0,
            )
            + 1
        )

    return DatasetEvidencePackage(
        package_version="1.0",
        table_name="dataset",
        row_count=100,
        column_count=5,
        intelligence={
            "capabilities": {
                "overview": True,
                "audit": True,
                "relationships": True,
            }
        },
        observations=observations,
        summary=EvidenceSummary(
            observation_count=len(observations),
            observation_types=list(
                type_counts.keys()
            ),
            observation_sources=list(
                source_counts.keys()
            ),
            source_counts=source_counts,
            type_counts=type_counts,
        ),
        metadata={},
    )


def create_execution_result(
    *,
    executed: list[int],
    partial: list[int] | None = None,
    skipped: list[int] | None = None,
) -> TierExecutionResult:
    store = ObservationStore()

    return TierExecutionResult(
        plan=ExecutionPlan(),
        executed_tiers=executed,
        partial_tiers=partial or [],
        skipped_tiers=skipped or [],
        observations=store,
        execution_details={},
    )


def create_observation(
    *,
    observation_id: str,
    observation_type: str,
    variables: list[str],
    statistic: dict,
    method: str = "test",
    source: str = "test_engine",
    sample_size: int = 100,
) -> Observation:
    return Observation(
        observation_id=observation_id,
        type=observation_type,
        variables=variables,
        statistic=statistic,
        sample_size=sample_size,
        method=method,
        source=source,
    )


def test_sufficient_evidence() -> None:
    observations = [
        create_observation(
            observation_id="obs_001",
            observation_type="correlation",
            variables=[
                "experience",
                "salary",
            ],
            statistic={
                "pearson_r": 0.74,
            },
            method="pearson",
            source="relationship_engine",
        )
    ]

    evidence = create_evidence(
        observations
    )

    execution = create_execution_result(
        executed=[2, 3, 5],
        skipped=[1, 4, 6],
    )

    decision = CriticValidator().evaluate(
        objective=(
            "Find relationships between "
            "experience and salary"
        ),
        evidence=evidence,
        execution_result=execution,
    )

    assert (
        decision.status
        == CriticStatus.SUFFICIENT
    )

    assert decision.replan_required is False
    assert decision.evidence_count == 1
    assert decision.contradiction_count == 0


def test_empty_evidence_is_insufficient() -> None:
    evidence = create_evidence(
        []
    )

    execution = create_execution_result(
        executed=[2],
    )

    decision = CriticValidator().evaluate(
        objective="Give me an overview",
        evidence=evidence,
        execution_result=execution,
    )

    assert (
        decision.status
        == CriticStatus.INSUFFICIENT
    )

    assert decision.replan_required is True

    assert (
        "at_least_one_observation"
        in decision.missing_requirements
    )


def test_query_requires_query_execution() -> None:
    observations = [
        create_observation(
            observation_id="obs_002",
            observation_type="distribution",
            variables=["salary"],
            statistic={
                "mean": 60000,
            },
        )
    ]

    evidence = create_evidence(
        observations
    )

    execution = create_execution_result(
        executed=[2, 3],
        skipped=[1, 4, 5, 6],
    )

    decision = CriticValidator().evaluate(
        objective="Run a SQL query to find rows",
        evidence=evidence,
        execution_result=execution,
    )

    assert (
        decision.status
        == CriticStatus.INSUFFICIENT
    )

    assert decision.replan_required is True

    assert (
        "query_execution"
        in decision.missing_requirements
    )


def test_repair_requires_repair_analysis() -> None:
    observations = [
        create_observation(
            observation_id="obs_003",
            observation_type="missing_values",
            variables=["age"],
            statistic={
                "null_count": 5,
            },
            source="quality_engine",
        )
    ]

    evidence = create_evidence(
        observations
    )

    execution = create_execution_result(
        executed=[2, 3],
        skipped=[1, 4, 5, 6],
    )

    decision = CriticValidator().evaluate(
        objective="Repair missing values",
        evidence=evidence,
        execution_result=execution,
    )

    assert (
        decision.status
        == CriticStatus.INSUFFICIENT
    )

    assert (
        "repair_analysis"
        in decision.missing_requirements
    )


def test_engineering_requires_engineering_analysis() -> None:
    observations = [
        create_observation(
            observation_id="obs_004",
            observation_type="distribution",
            variables=["salary"],
            statistic={
                "mean": 60000,
            },
        )
    ]

    evidence = create_evidence(
        observations
    )

    execution = create_execution_result(
        executed=[2, 3],
        skipped=[1, 4, 5, 6],
    )

    decision = CriticValidator().evaluate(
        objective="Migrate this dataset to a new database",
        evidence=evidence,
        execution_result=execution,
    )

    assert (
        decision.status
        == CriticStatus.INSUFFICIENT
    )

    assert (
        "engineering_analysis"
        in decision.missing_requirements
    )


def test_partial_execution_is_recorded() -> None:
    observations = [
        create_observation(
            observation_id="obs_005",
            observation_type="distribution",
            variables=["salary"],
            statistic={
                "mean": 60000,
            },
        )
    ]

    evidence = create_evidence(
        observations
    )

    execution = create_execution_result(
        executed=[2, 3],
        partial=[4],
        skipped=[1, 5, 6],
    )

    decision = CriticValidator().evaluate(
        objective="Give me an overview",
        evidence=evidence,
        execution_result=execution,
    )

    assert (
        decision.status
        == CriticStatus.SUFFICIENT
    )

    assert decision.partial_tiers == [4]

    assert any(
        "partially executed" in reason
        for reason in decision.reasons
    )


def test_contradictory_evidence_is_detected() -> None:
    observations = [
        create_observation(
            observation_id="obs_006",
            observation_type="correlation",
            variables=[
                "experience",
                "salary",
            ],
            statistic={
                "pearson_r": 0.82,
            },
            method="pearson",
            source="relationship_engine",
        ),
        create_observation(
            observation_id="obs_007",
            observation_type="correlation",
            variables=[
                "experience",
                "salary",
            ],
            statistic={
                "pearson_r": -0.79,
            },
            method="pearson",
            source="relationship_engine",
        ),
    ]

    evidence = create_evidence(
        observations
    )

    execution = create_execution_result(
        executed=[2, 3, 5],
    )

    decision = CriticValidator().evaluate(
        objective=(
            "Find relationships between "
            "experience and salary"
        ),
        evidence=evidence,
        execution_result=execution,
    )

    assert (
        decision.status
        == CriticStatus.CONTRADICTORY
    )

    assert decision.replan_required is True
    assert decision.contradiction_count == 1


def test_normal_different_methods_are_not_contradictory() -> None:
    observations = [
        create_observation(
            observation_id="obs_008",
            observation_type="correlation",
            variables=[
                "experience",
                "salary",
            ],
            statistic={
                "pearson_r": 0.82,
            },
            method="pearson",
            source="relationship_engine",
        ),
        create_observation(
            observation_id="obs_009",
            observation_type="correlation",
            variables=[
                "experience",
                "salary",
            ],
            statistic={
                "spearman_rho": 0.79,
            },
            method="spearman",
            source="relationship_engine",
        ),
    ]

    evidence = create_evidence(
        observations
    )

    execution = create_execution_result(
        executed=[2, 3, 5],
    )

    decision = CriticValidator().evaluate(
        objective=(
            "Find relationships between "
            "experience and salary"
        ),
        evidence=evidence,
        execution_result=execution,
    )

    assert (
        decision.status
        == CriticStatus.SUFFICIENT
    )

    assert decision.contradiction_count == 0


def test_decision_validation() -> None:
    observations = [
        create_observation(
            observation_id="obs_010",
            observation_type="distribution",
            variables=["salary"],
            statistic={
                "mean": 60000,
            },
        )
    ]

    evidence = create_evidence(
        observations
    )

    execution = create_execution_result(
        executed=[2],
    )

    decision = CriticValidator().evaluate(
        objective="Give me an overview",
        evidence=evidence,
        execution_result=execution,
    )

    CriticValidator.validate_decision(
        decision
    )


def main() -> None:
    print("=" * 60)
    print("NEXUS-6 CRITIC / VALIDATOR TEST")
    print("=" * 60)

    test_sufficient_evidence()
    print("PASS: Sufficient evidence")

    test_empty_evidence_is_insufficient()
    print("PASS: Empty evidence detection")

    test_query_requires_query_execution()
    print("PASS: Query requirement validation")

    test_repair_requires_repair_analysis()
    print("PASS: Repair requirement validation")

    test_engineering_requires_engineering_analysis()
    print("PASS: Engineering requirement validation")

    test_partial_execution_is_recorded()
    print("PASS: Partial execution tracking")

    test_contradictory_evidence_is_detected()
    print("PASS: Contradiction detection")

    test_normal_different_methods_are_not_contradictory()
    print("PASS: Method-aware contradiction handling")

    test_decision_validation()
    print("PASS: Decision model validation")

    print()
    print("=" * 60)
    print("ALL NEXUS-6 CRITIC / VALIDATOR TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()