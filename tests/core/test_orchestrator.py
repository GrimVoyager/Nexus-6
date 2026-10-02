from __future__ import annotations

from pathlib import Path
import tempfile

from backend.core.evidence import (
    DatasetEvidenceBuilder,
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


GENERAL_CSV_CONTENT = """Age,Height,Weight,Score
20,170,65,72
21,175,70,75
22,168,62,81
23,180,78,88
24,172,69,91
25,177,73,79
"""


def create_dataset(
    content: str = CSV_CONTENT,
) -> Path:
    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix="nexus6_orchestrator_"
        )
    )

    dataset_path = (
        temporary_directory
        / "dataset.csv"
    )

    dataset_path.write_text(
        content,
        encoding="utf-8",
    )

    return dataset_path


def build_evidence(
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

    collector = ObservationCollector(
        quality_analyzer=quality_analyzer,
        relationship_analyzer=relationship_analyzer,
        statistical_analyzer=statistical_analyzer,
    )

    builder = DatasetEvidenceBuilder(
        intelligence=intelligence,
        observation_collector=collector,
    )

    return engine, builder.build()


def test_orchestration_request_rejects_empty_objective() -> None:
    try:
        OrchestrationRequest(
            objective="   "
        )
    except ValueError as error:
        assert "objective" in str(error)
    else:
        raise AssertionError(
            "Expected empty objective to be rejected."
        )


def test_hr_domain_detection() -> None:
    dataset_path = create_dataset()

    engine, evidence = build_evidence(
        dataset_path
    )

    try:
        orchestrator = NexusOrchestrator()

        plan = orchestrator.orchestrate(
            evidence=evidence,
            request=OrchestrationRequest(
                objective="Analyze employee performance."
            ),
        )

        assert (
            plan.detected_domain
            == "hr_workforce"
        )

        assert plan.overlay is True

    finally:
        engine.close()


def test_general_domain_when_signals_are_insufficient() -> None:
    dataset_path = create_dataset(
        GENERAL_CSV_CONTENT
    )

    engine, evidence = build_evidence(
        dataset_path
    )

    try:
        orchestrator = NexusOrchestrator()

        plan = orchestrator.orchestrate(
            evidence=evidence,
            request=OrchestrationRequest(
                objective="Analyze the dataset."
            ),
        )

        assert (
            plan.detected_domain
            == "general"
        )

        assert plan.overlay is False

    finally:
        engine.close()


def test_snapshot_objective() -> None:
    dataset_path = create_dataset()

    engine, evidence = build_evidence(
        dataset_path
    )

    try:
        orchestrator = NexusOrchestrator()

        plan = orchestrator.orchestrate(
            evidence=evidence,
            request=OrchestrationRequest(
                objective="Give me an overview of this dataset."
            ),
        )

        overview = plan.get_tier_decision(2)
        intelligence = plan.get_tier_decision(5)
        engineering = plan.get_tier_decision(6)

        assert overview is not None
        assert intelligence is not None
        assert engineering is not None

        assert overview.status == "EXECUTE"
        assert intelligence.status == "EXECUTE"
        assert engineering.status == "NOT_APPLICABLE"

    finally:
        engine.close()


def test_query_objective() -> None:
    dataset_path = create_dataset()

    engine, evidence = build_evidence(
        dataset_path
    )

    try:
        orchestrator = NexusOrchestrator()

        plan = orchestrator.orchestrate(
            evidence=evidence,
            request=OrchestrationRequest(
                objective="Show me employees with salary above 70000."
            ),
        )

        query = plan.get_tier_decision(1)

        assert query is not None
        assert query.status == "EXECUTE"

    finally:
        engine.close()


def test_deep_analysis_objective() -> None:
    dataset_path = create_dataset()

    engine, evidence = build_evidence(
        dataset_path
    )

    try:
        orchestrator = NexusOrchestrator()

        plan = orchestrator.orchestrate(
            evidence=evidence,
            request=OrchestrationRequest(
                objective=(
                    "Perform a deep analysis of employee "
                    "performance and relationships."
                )
            ),
        )

        intelligence = plan.get_tier_decision(5)

        assert intelligence is not None
        assert intelligence.status == "EXECUTE"

    finally:
        engine.close()


def test_repair_objective_is_partial() -> None:
    dataset_path = create_dataset()

    engine, evidence = build_evidence(
        dataset_path
    )

    try:
        orchestrator = NexusOrchestrator()

        plan = orchestrator.orchestrate(
            evidence=evidence,
            request=OrchestrationRequest(
                objective=(
                    "Clean and repair the employee dataset."
                )
            ),
        )

        repair = plan.get_tier_decision(4)

        assert repair is not None
        assert repair.status == "PARTIAL"

    finally:
        engine.close()


def test_engineering_objective() -> None:
    dataset_path = create_dataset()

    engine, evidence = build_evidence(
        dataset_path
    )

    try:
        orchestrator = NexusOrchestrator()

        plan = orchestrator.orchestrate(
            evidence=evidence,
            request=OrchestrationRequest(
                objective=(
                    "Prepare this dataset for a database "
                    "migration and ETL pipeline."
                )
            ),
        )

        engineering = plan.get_tier_decision(6)

        assert engineering is not None
        assert engineering.status == "EXECUTE"

    finally:
        engine.close()


def test_target_state_without_explicit_target() -> None:
    dataset_path = create_dataset()

    engine, evidence = build_evidence(
        dataset_path
    )

    try:
        orchestrator = NexusOrchestrator()

        plan = orchestrator.orchestrate(
            evidence=evidence,
            request=OrchestrationRequest(
                objective=(
                    "Analyze this employee dataset."
                )
            ),
        )

        assert plan.target is None

        assert (
            "Target state is 'TARGET_UNCERTAIN'"
            in plan.reasoning
        )

    finally:
        engine.close()


def test_target_found_when_named_in_objective() -> None:
    dataset_path = create_dataset()

    engine, evidence = build_evidence(
        dataset_path
    )

    try:
        orchestrator = NexusOrchestrator()

        plan = orchestrator.orchestrate(
            evidence=evidence,
            request=OrchestrationRequest(
                objective=(
                    "Predict Performance using the employee data."
                )
            ),
        )

        assert (
            plan.target
            == "Performance"
        )

        assert (
            "Target state is 'TARGET_FOUND'"
            in plan.reasoning
        )

    finally:
        engine.close()


def test_all_six_tiers_are_present() -> None:
    dataset_path = create_dataset()

    engine, evidence = build_evidence(
        dataset_path
    )

    try:
        orchestrator = NexusOrchestrator()

        plan = orchestrator.orchestrate(
            evidence=evidence,
            request=OrchestrationRequest(
                objective="Analyze the dataset."
            ),
        )

        tiers = [
            int(decision.tier)
            for decision in plan.tier_decisions
        ]

        assert tiers == [
            1,
            2,
            3,
            4,
            5,
            6,
        ]

    finally:
        engine.close()


def test_orchestrator_produces_valid_execution_plan() -> None:
    dataset_path = create_dataset()

    engine, evidence = build_evidence(
        dataset_path
    )

    try:
        orchestrator = NexusOrchestrator()

        plan = orchestrator.orchestrate(
            evidence=evidence,
            request=OrchestrationRequest(
                objective="Analyze the dataset."
            ),
        )

        assert plan.selected_tiers == [
            1,
            2,
            3,
            4,
            5,
            6,
        ]

        assert plan.reasoning

        serialized = plan.to_json()

        assert '"tier_decisions"' in serialized
        assert '"detected_domain"' in serialized

    finally:
        engine.close()


def main() -> None:
    print("=" * 60)
    print("NEXUS-6 ORCHESTRATOR TEST")
    print("=" * 60)

    test_orchestration_request_rejects_empty_objective()
    print("PASS: Objective validation")

    test_hr_domain_detection()
    print("PASS: HR domain detection")

    test_general_domain_when_signals_are_insufficient()
    print("PASS: General-domain fallback")

    test_snapshot_objective()
    print("PASS: Snapshot objective routing")

    test_query_objective()
    print("PASS: Query objective routing")

    test_deep_analysis_objective()
    print("PASS: Deep-analysis routing")

    test_repair_objective_is_partial()
    print("PASS: Repair safety routing")

    test_engineering_objective()
    print("PASS: Engineering routing")

    test_target_state_without_explicit_target()
    print("PASS: Target uncertainty")

    test_target_found_when_named_in_objective()
    print("PASS: Explicit target detection")

    test_all_six_tiers_are_present()
    print("PASS: Six-tier coverage")

    test_orchestrator_produces_valid_execution_plan()
    print("PASS: ExecutionPlan generation")

    print()
    print("=" * 60)
    print("ALL NEXUS-6 ORCHESTRATOR TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()