from __future__ import annotations

from pathlib import Path
import tempfile

from backend.core.evidence import (
    DatasetEvidenceBuilder,
)
from backend.core.evidence_merger import (
    EvidenceMerger,
)
from backend.core.execution_observation_adapter import (
    ExecutionObservationAdapter,
)
from backend.core.executor import (
    TierExecutor,
)
from backend.core.intelligence import (
    DatasetIntelligence,
)
from backend.core.loader import DataEngine
from backend.core.observation_collector import (
    ObservationCollector,
)
from backend.core.orchestrator import (
    NexusOrchestrator,
    OrchestrationRequest,
)
from backend.core.quality import (
    DataQualityAnalyzer,
)
from backend.core.relationships import (
    RelationshipAnalyzer,
)
from backend.core.statistics import (
    StatisticalAnalyzer,
)


CSV_CONTENT = """Employee_Name,Department,Experience,Salary,Performance
Alice,Engineering,2,50000,72
Bob,Engineering,3,52000,75
Charlie,HR,5,60000,81
David,Sales,7,68000,88
Eve,Sales,8,72000,91
Frank,Engineering,4,58000,79
Grace,HR,6,64000,84
Henry,Sales,10,85000,94
Iris,Engineering,3,54000,76
Jack,HR,5,62000,82
Kate,Sales,9,78000,92
Leo,Engineering,4,56000,78
"""


def create_dataset() -> Path:
    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix="nexus6_integration_"
        )
    )

    dataset_path = (
        temporary_directory
        / "employees.csv"
    )

    dataset_path.write_text(
        CSV_CONTENT,
        encoding="utf-8",
    )

    return dataset_path


def build_components(
    dataset_path: Path,
):
    engine = DataEngine()

    engine.read_file(
        dataset_path,
        table_name="dataset",
    )

    intelligence = DatasetIntelligence(
        engine=engine,
        table_name="dataset",
    )

    quality_analyzer = DataQualityAnalyzer(
        engine=engine,
        table_name="dataset",
    )

    relationship_analyzer = RelationshipAnalyzer(
        engine=engine,
        table_name="dataset",
    )

    statistical_analyzer = StatisticalAnalyzer(
        engine=engine,
        table_name="dataset",
    )

    observation_collector = ObservationCollector(
        quality_analyzer=quality_analyzer,
        relationship_analyzer=relationship_analyzer,
        statistical_analyzer=statistical_analyzer,
    )

    evidence_builder = DatasetEvidenceBuilder(
        intelligence=intelligence,
        observation_collector=observation_collector,
    )

    evidence = evidence_builder.build()

    executor = TierExecutor(
        intelligence=intelligence,
        quality_analyzer=quality_analyzer,
        relationship_analyzer=relationship_analyzer,
        statistical_analyzer=statistical_analyzer,
        observation_collector=observation_collector,
    )

    return (
        engine,
        evidence,
        executor,
    )


def test_execution_to_evidence_pipeline() -> None:
    dataset_path = create_dataset()

    engine, evidence, executor = build_components(
        dataset_path
    )

    try:
        original_observation_count = (
            evidence.summary.observation_count
        )

        orchestrator = NexusOrchestrator()

        plan = orchestrator.orchestrate(
            evidence=evidence,
            request=OrchestrationRequest(
                objective=(
                    "Perform a deep analysis of "
                    "employee performance and relationships."
                )
            ),
        )

        execution_result = executor.execute(
            evidence=evidence,
            plan=plan,
        )

        assert len(
            execution_result.executed_tiers
        ) > 0

        adapter = ExecutionObservationAdapter()

        execution_observations = adapter.collect(
            execution_result
        )

        assert (
            execution_observations.count()
            > 0
        )

        for observation in (
            execution_observations.all()
        ):
            assert (
                observation.type
                == "tier_execution"
            )

            assert (
                observation.source
                == "execution_observation_adapter"
            )

            assert (
                observation.method
                == "deterministic_execution_adapter"
            )

        execution_result.observations = (
            execution_observations
        )

        merger = EvidenceMerger()

        merged_evidence = merger.merge(
            original=evidence,
            execution_result=execution_result,
        )

        assert (
            merged_evidence.summary.observation_count
            == (
                original_observation_count
                + execution_observations.count()
            )
        )

        assert (
            merged_evidence.summary.observation_count
            > original_observation_count
        )

        execution_source_count = (
            merged_evidence.summary.source_counts.get(
                "execution_observation_adapter",
                0,
            )
        )

        assert (
            execution_source_count
            == execution_observations.count()
        )

    finally:
        engine.close()


def test_original_evidence_remains_unchanged() -> None:
    dataset_path = create_dataset()

    engine, evidence, executor = build_components(
        dataset_path
    )

    try:
        original_json = evidence.to_json()
        original_count = (
            evidence.summary.observation_count
        )

        orchestrator = NexusOrchestrator()

        plan = orchestrator.orchestrate(
            evidence=evidence,
            request=OrchestrationRequest(
                objective=(
                    "Perform a deep analysis of "
                    "employee performance."
                )
            ),
        )

        execution_result = executor.execute(
            evidence=evidence,
            plan=plan,
        )

        adapter = ExecutionObservationAdapter()

        execution_observations = adapter.collect(
            execution_result
        )

        execution_result.observations = (
            execution_observations
        )

        merger = EvidenceMerger()

        merged_evidence = merger.merge(
            original=evidence,
            execution_result=execution_result,
        )

        assert (
            evidence.to_json()
            == original_json
        )

        assert (
            evidence.summary.observation_count
            == original_count
        )

        assert (
            merged_evidence is not evidence
        )

    finally:
        engine.close()


def test_pipeline_merge_is_idempotent() -> None:
    dataset_path = create_dataset()

    engine, evidence, executor = build_components(
        dataset_path
    )

    try:
        orchestrator = NexusOrchestrator()

        plan = orchestrator.orchestrate(
            evidence=evidence,
            request=OrchestrationRequest(
                objective=(
                    "Perform a deep analysis of "
                    "employee performance."
                )
            ),
        )

        execution_result = executor.execute(
            evidence=evidence,
            plan=plan,
        )

        adapter = ExecutionObservationAdapter()

        execution_observations = adapter.collect(
            execution_result
        )

        execution_result.observations = (
            execution_observations
        )

        merger = EvidenceMerger()

        first_merge = merger.merge(
            original=evidence,
            execution_result=execution_result,
        )

        second_merge = merger.merge(
            original=first_merge,
            execution_result=execution_result,
        )

        assert (
            second_merge.summary.observation_count
            == first_merge.summary.observation_count
        )

        assert (
            second_merge.to_json()
            == first_merge.to_json()
        )

    finally:
        engine.close()


def test_execution_provenance_survives_merge() -> None:
    dataset_path = create_dataset()

    engine, evidence, executor = build_components(
        dataset_path
    )

    try:
        orchestrator = NexusOrchestrator()

        plan = orchestrator.orchestrate(
            evidence=evidence,
            request=OrchestrationRequest(
                objective=(
                    "Perform a deep analysis of "
                    "employee performance."
                )
            ),
        )

        execution_result = executor.execute(
            evidence=evidence,
            plan=plan,
        )

        adapter = ExecutionObservationAdapter()

        execution_observations = adapter.collect(
            execution_result
        )

        execution_result.observations = (
            execution_observations
        )

        merger = EvidenceMerger()

        merged_evidence = merger.merge(
            original=evidence,
            execution_result=execution_result,
        )

        merged_execution_observations = [
            observation
            for observation in merged_evidence.observations
            if (
                observation.source
                == "execution_observation_adapter"
            )
        ]

        assert (
            len(merged_execution_observations)
            == execution_observations.count()
        )

        for observation in (
            merged_execution_observations
        ):
            assert (
                observation.metadata[
                    "execution_status"
                ]
                in {
                    "EXECUTED",
                    "PARTIAL",
                }
            )

            assert (
                observation.metadata["tier"]
                in (
                    execution_result.executed_tiers
                    + execution_result.partial_tiers
                )
            )

    finally:
        engine.close()


def main() -> None:
    print("=" * 60)
    print(
        "NEXUS-6 EXECUTION → EVIDENCE "
        "PIPELINE INTEGRATION TEST"
    )
    print("=" * 60)

    test_execution_to_evidence_pipeline()
    print(
        "PASS: Executor → Adapter → Evidence Merger"
    )

    test_original_evidence_remains_unchanged()
    print(
        "PASS: Original evidence remains unchanged"
    )

    test_pipeline_merge_is_idempotent()
    print(
        "PASS: Pipeline merge is idempotent"
    )

    test_execution_provenance_survives_merge()
    print(
        "PASS: Execution provenance survives merge"
    )

    print()
    print("=" * 60)
    print(
        "ALL EXECUTION → EVIDENCE "
        "INTEGRATION TESTS PASSED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()