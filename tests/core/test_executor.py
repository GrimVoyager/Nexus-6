from __future__ import annotations

from pathlib import Path
import tempfile

from backend.core.evidence import (
    DatasetEvidenceBuilder,
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
from backend.core.planner import (
    AnalysisTier,
    ExecutionPlan,
    TierDecision,
    TierStatus,
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
            prefix="nexus6_executor_"
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


def build_plan(
    evidence,
    objective: str,
) -> ExecutionPlan:
    orchestrator = NexusOrchestrator()

    return orchestrator.orchestrate(
        evidence=evidence,
        request=OrchestrationRequest(
            objective=objective
        ),
    )


def test_overview_execution() -> None:
    dataset_path = create_dataset()

    engine, evidence, executor = build_components(
        dataset_path
    )

    try:
        plan = build_plan(
            evidence,
            "Give me an overview of this dataset.",
        )

        result = executor.execute(
            evidence=evidence,
            plan=plan,
        )

        assert 2 in result.executed_tiers

        assert (
            result.execution_details["2"]["status"]
            == "EXECUTED"
        )

        assert (
            result.execution_details["2"]["executed"]
            is True
        )

    finally:
        engine.close()


def test_audit_execution() -> None:
    dataset_path = create_dataset()

    engine, evidence, executor = build_components(
        dataset_path
    )

    try:
        decisions = [
            TierDecision(
                tier=AnalysisTier.QUERY,
                name="Query",
                status=TierStatus.NOT_APPLICABLE,
                reason="Not required.",
            ),
            TierDecision(
                tier=AnalysisTier.OVERVIEW,
                name="Overview",
                status=TierStatus.NOT_APPLICABLE,
                reason="Not required.",
            ),
            TierDecision(
                tier=AnalysisTier.AUDIT,
                name="Audit",
                status=TierStatus.EXECUTE,
                reason="Audit requested.",
            ),
            TierDecision(
                tier=AnalysisTier.REPAIR,
                name="Repair",
                status=TierStatus.NOT_APPLICABLE,
                reason="Not required.",
            ),
            TierDecision(
                tier=AnalysisTier.INTELLIGENCE,
                name="Intelligence",
                status=TierStatus.NOT_APPLICABLE,
                reason="Not required.",
            ),
            TierDecision(
                tier=AnalysisTier.ENGINEERING,
                name="Engineering",
                status=TierStatus.NOT_APPLICABLE,
                reason="Not required.",
            ),
        ]

        plan = ExecutionPlan(
            tier_decisions=decisions,
            detected_domain="general",
        )

        result = executor.execute(
            evidence=evidence,
            plan=plan,
        )

        assert result.executed_tiers == [3]

        audit_result = (
            result.execution_details["3"]
        )

        assert (
            audit_result["status"]
            == "EXECUTED"
        )

        assert "quality_report" in audit_result
        assert "statistical_report" in audit_result

    finally:
        engine.close()


def test_intelligence_execution() -> None:
    dataset_path = create_dataset()

    engine, evidence, executor = build_components(
        dataset_path
    )

    try:
        plan = build_plan(
            evidence,
            (
                "Perform a deep analysis of employee "
                "performance and relationships."
            ),
        )

        result = executor.execute(
            evidence=evidence,
            plan=plan,
        )

        assert 5 in result.executed_tiers

        intelligence_result = (
            result.execution_details["5"]
        )

        assert (
            intelligence_result["status"]
            == "EXECUTED"
        )

        assert "relationship_report" in (
            intelligence_result
        )

        assert "statistical_report" in (
            intelligence_result
        )

    finally:
        engine.close()


def test_not_applicable_tiers_are_not_executed() -> None:
    dataset_path = create_dataset()

    engine, evidence, executor = build_components(
        dataset_path
    )

    try:
        plan = build_plan(
            evidence,
            "Give me an overview of this dataset.",
        )

        result = executor.execute(
            evidence=evidence,
            plan=plan,
        )

        assert 6 in result.skipped_tiers

        engineering_result = (
            result.execution_details["6"]
        )

        assert (
            engineering_result["status"]
            == "NOT_EXECUTED"
        )

    finally:
        engine.close()


def test_repair_never_mutates_source() -> None:
    dataset_path = create_dataset()

    original_content = dataset_path.read_text(
        encoding="utf-8"
    )

    engine, evidence, executor = build_components(
        dataset_path
    )

    try:
        plan = build_plan(
            evidence,
            "Clean and repair the employee dataset.",
        )

        result = executor.execute(
            evidence=evidence,
            plan=plan,
        )

        assert 4 in result.partial_tiers

        repair_result = (
            result.execution_details["4"]
        )

        assert (
            repair_result["mutation_performed"]
            is False
        )

    finally:
        engine.close()

    final_content = dataset_path.read_text(
        encoding="utf-8"
    )

    assert final_content == original_content


def test_engineering_is_planned_not_executed() -> None:
    dataset_path = create_dataset()

    engine, evidence, executor = build_components(
        dataset_path
    )

    try:
        plan = build_plan(
            evidence,
            (
                "Prepare this dataset for a database "
                "migration and ETL pipeline."
            ),
        )

        result = executor.execute(
            evidence=evidence,
            plan=plan,
        )

        assert 6 in result.executed_tiers

        engineering_result = (
            result.execution_details["6"]
        )

        assert (
            engineering_result["status"]
            == "PLANNED"
        )

        assert (
            engineering_result["executed"]
            is False
        )

    finally:
        engine.close()


def test_query_is_not_executed_without_query_engine() -> None:
    dataset_path = create_dataset()

    engine, evidence, executor = build_components(
        dataset_path
    )

    try:
        decisions = [
            TierDecision(
                tier=AnalysisTier.QUERY,
                name="Query",
                status=TierStatus.EXECUTE,
                reason="Query requested.",
            ),
            TierDecision(
                tier=AnalysisTier.OVERVIEW,
                name="Overview",
                status=TierStatus.NOT_APPLICABLE,
                reason="Not required.",
            ),
            TierDecision(
                tier=AnalysisTier.AUDIT,
                name="Audit",
                status=TierStatus.NOT_APPLICABLE,
                reason="Not required.",
            ),
            TierDecision(
                tier=AnalysisTier.REPAIR,
                name="Repair",
                status=TierStatus.NOT_APPLICABLE,
                reason="Not required.",
            ),
            TierDecision(
                tier=AnalysisTier.INTELLIGENCE,
                name="Intelligence",
                status=TierStatus.NOT_APPLICABLE,
                reason="Not required.",
            ),
            TierDecision(
                tier=AnalysisTier.ENGINEERING,
                name="Engineering",
                status=TierStatus.NOT_APPLICABLE,
                reason="Not required.",
            ),
        ]

        plan = ExecutionPlan(
            tier_decisions=decisions,
            detected_domain="general",
            primary_sql=(
                "SELECT * FROM dataset"
            ),
        )

        result = executor.execute(
            evidence=evidence,
            plan=plan,
        )

        assert result.executed_tiers == [1]

        query_result = (
            result.execution_details["1"]
        )

        assert (
            query_result["status"]
            == "PLANNED"
        )

        assert (
            query_result["executed"]
            is False
        )

    finally:
        engine.close()


def test_execution_result_contains_valid_structure() -> None:
    dataset_path = create_dataset()

    engine, evidence, executor = build_components(
        dataset_path
    )

    try:
        plan = build_plan(
            evidence,
            "Perform a deep analysis.",
        )

        result = executor.execute(
            evidence=evidence,
            plan=plan,
        )

        result_dict = result.to_dict()

        assert "executed_tiers" in result_dict
        assert "partial_tiers" in result_dict
        assert "skipped_tiers" in result_dict
        assert "observation_count" in result_dict
        assert "execution_details" in result_dict
        assert "observations" in result_dict

    finally:
        engine.close()


def main() -> None:
    print("=" * 60)
    print("NEXUS-6 TIER EXECUTOR TEST")
    print("=" * 60)

    test_overview_execution()
    print("PASS: Overview execution")

    test_audit_execution()
    print("PASS: Audit execution")

    test_intelligence_execution()
    print("PASS: Intelligence execution")

    test_not_applicable_tiers_are_not_executed()
    print("PASS: NOT_APPLICABLE protection")

    test_repair_never_mutates_source()
    print("PASS: Repair mutation protection")

    test_engineering_is_planned_not_executed()
    print("PASS: Engineering execution boundary")

    test_query_is_not_executed_without_query_engine()
    print("PASS: Query execution boundary")

    test_execution_result_contains_valid_structure()
    print("PASS: Execution result structure")

    print()
    print("=" * 60)
    print("ALL NEXUS-6 TIER EXECUTOR TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()