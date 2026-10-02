from __future__ import annotations
from backend.core.planner import ExecutionPlan
from backend.core.evidence import (
    DatasetEvidencePackage,
    EvidenceSummary,
)
from backend.core.evidence_merger import EvidenceMerger
from backend.core.executor import TierExecutionResult
from backend.core.observation import (
    Observation,
    ObservationStore,
)


def create_original_evidence() -> DatasetEvidencePackage:
    observation = Observation(
        observation_id="obs_original_001",
        type="distribution",
        variables=["Salary"],
        statistic={
            "mean": 60000.0,
        },
        sample_size=10,
        method="descriptive_statistics",
        source="statistics_engine",
    )

    return DatasetEvidencePackage(
        package_version="1.0",
        table_name="dataset",
        row_count=10,
        column_count=3,
        intelligence={
            "capabilities": {
                "overview": True,
            }
        },
        observations=[
            observation,
        ],
        summary=EvidenceSummary(
            observation_count=1,
            observation_types=[
                "distribution",
            ],
            observation_sources=[
                "statistics_engine",
            ],
            source_counts={
                "statistics_engine": 1,
            },
            type_counts={
                "distribution": 1,
            },
        ),
        metadata={
            "original": True,
        },
    )


def create_execution_result(
    observations: list[Observation],
) -> TierExecutionResult:
    store = ObservationStore()

    store.add_many(
        observations
    )

    return TierExecutionResult(
        plan=ExecutionPlan(),
        executed_tiers=[3],
        partial_tiers=[],
        skipped_tiers=[1, 2, 4, 5, 6],
        observations=store,
        execution_details={
            "3": {
                "status": "EXECUTED",
            }
        },
    )


def test_new_observation_is_merged() -> None:
    original = create_original_evidence()

    new_observation = Observation(
        observation_id="obs_new_001",
        type="missing_values",
        variables=["Age"],
        statistic={
            "null_count": 2,
        },
        sample_size=10,
        method="null_analysis",
        source="quality_engine",
    )

    execution_result = create_execution_result(
        [new_observation]
    )

    merger = EvidenceMerger()

    updated = merger.merge(
        original=original,
        execution_result=execution_result,
    )

    assert len(updated.observations) == 2

    assert (
        updated.get_observations_by_type(
            "missing_values"
        )[0].observation_id
        == "obs_new_001"
    )


def test_original_evidence_is_preserved() -> None:
    original = create_original_evidence()

    new_observation = Observation(
        observation_id="obs_new_002",
        type="correlation",
        variables=[
            "Experience",
            "Salary",
        ],
        statistic={
            "pearson": 0.82,
        },
        sample_size=10,
        method="pearson",
        source="relationship_engine",
    )

    execution_result = create_execution_result(
        [new_observation]
    )

    updated = EvidenceMerger().merge(
        original=original,
        execution_result=execution_result,
    )

    original_observation = (
        updated.get_observations_by_type(
            "distribution"
        )[0]
    )

    assert (
        original_observation.observation_id
        == "obs_original_001"
    )


def test_identical_duplicate_is_idempotent() -> None:
    original = create_original_evidence()

    duplicate = Observation(
        observation_id="obs_original_001",
        type="distribution",
        variables=["Salary"],
        statistic={
            "mean": 60000.0,
        },
        sample_size=10,
        method="descriptive_statistics",
        source="statistics_engine",
    )

    execution_result = create_execution_result(
        [duplicate]
    )

    updated = EvidenceMerger().merge(
        original=original,
        execution_result=execution_result,
    )

    assert len(updated.observations) == 1
    assert updated.summary.observation_count == 1


def test_conflicting_duplicate_is_rejected() -> None:
    original = create_original_evidence()

    conflicting = Observation(
        observation_id="obs_original_001",
        type="distribution",
        variables=["Salary"],
        statistic={
            "mean": 999999.0,
        },
        sample_size=10,
        method="descriptive_statistics",
        source="statistics_engine",
    )

    execution_result = create_execution_result(
        [conflicting]
    )

    merger = EvidenceMerger()

    try:
        merger.merge(
            original=original,
            execution_result=execution_result,
        )
    except ValueError as error:
        assert (
            "different content"
            in str(error)
        )
    else:
        raise AssertionError(
            "Conflicting duplicate observation "
            "was not rejected."
        )


def test_summary_is_rebuilt() -> None:
    original = create_original_evidence()

    new_observations = [
        Observation(
            observation_id="obs_new_003",
            type="missing_values",
            variables=["Age"],
            statistic={
                "null_count": 1,
            },
            sample_size=10,
            method="null_analysis",
            source="quality_engine",
        ),
        Observation(
            observation_id="obs_new_004",
            type="correlation",
            variables=[
                "Experience",
                "Salary",
            ],
            statistic={
                "pearson": 0.82,
            },
            sample_size=10,
            method="pearson",
            source="relationship_engine",
        ),
    ]

    execution_result = create_execution_result(
        new_observations
    )

    updated = EvidenceMerger().merge(
        original=original,
        execution_result=execution_result,
    )

    assert updated.summary.observation_count == 3

    assert (
        updated.summary.source_counts[
            "statistics_engine"
        ]
        == 1
    )

    assert (
        updated.summary.source_counts[
            "quality_engine"
        ]
        == 1
    )

    assert (
        updated.summary.source_counts[
            "relationship_engine"
        ]
        == 1
    )

    assert (
        updated.summary.type_counts[
            "distribution"
        ]
        == 1
    )

    assert (
        updated.summary.type_counts[
            "missing_values"
        ]
        == 1
    )

    assert (
        updated.summary.type_counts[
            "correlation"
        ]
        == 1
    )


def test_execution_provenance_is_recorded() -> None:
    original = create_original_evidence()

    execution_result = create_execution_result(
        []
    )

    updated = EvidenceMerger().merge(
        original=original,
        execution_result=execution_result,
    )

    merge_metadata = updated.metadata[
        "evidence_merge"
    ]

    assert (
        merge_metadata["performed"]
        is True
    )

    assert (
        merge_metadata[
            "original_observation_count"
        ]
        == 1
    )

    assert (
        merge_metadata[
            "execution_observation_count"
        ]
        == 0
    )

    assert (
        merge_metadata[
            "merged_observation_count"
        ]
        == 1
    )

    assert (
        merge_metadata["executed_tiers"]
        == [3]
    )


def test_original_package_is_not_mutated() -> None:
    original = create_original_evidence()

    original_count = len(
        original.observations
    )

    new_observation = Observation(
        observation_id="obs_new_005",
        type="outliers",
        variables=["Salary"],
        statistic={
            "outlier_count": 1,
        },
        sample_size=10,
        method="iqr_1_5_rule",
        source="statistics_engine",
    )

    execution_result = create_execution_result(
        [new_observation]
    )

    updated = EvidenceMerger().merge(
        original=original,
        execution_result=execution_result,
    )

    assert len(
        original.observations
    ) == original_count

    assert len(
        updated.observations
    ) == original_count + 1


def test_updated_package_serializes() -> None:
    original = create_original_evidence()

    execution_result = create_execution_result(
        []
    )

    updated = EvidenceMerger().merge(
        original=original,
        execution_result=execution_result,
    )

    serialized = updated.to_json()

    assert isinstance(
        serialized,
        str,
    )

    assert "observation_count" in serialized


def main() -> None:
    print("=" * 60)
    print("NEXUS-6 EVIDENCE MERGER TEST")
    print("=" * 60)

    test_new_observation_is_merged()
    print("PASS: New observation merge")

    test_original_evidence_is_preserved()
    print("PASS: Original evidence preservation")

    test_identical_duplicate_is_idempotent()
    print("PASS: Idempotent duplicate handling")

    test_conflicting_duplicate_is_rejected()
    print("PASS: Conflicting duplicate protection")

    test_summary_is_rebuilt()
    print("PASS: Evidence summary rebuild")

    test_execution_provenance_is_recorded()
    print("PASS: Execution provenance")

    test_original_package_is_not_mutated()
    print("PASS: Original package immutability")

    test_updated_package_serializes()
    print("PASS: Updated package serialization")

    print()
    print("=" * 60)
    print("ALL NEXUS-6 EVIDENCE MERGER TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()