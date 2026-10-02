from __future__ import annotations

import json
import tempfile
from pathlib import Path

from backend.core.evidence import (
    DatasetEvidenceBuilder,
    DatasetEvidencePackage,
)
from backend.core.intelligence import (
    DatasetIntelligence,
)
from backend.core.loader import DataEngine
from backend.core.observation_collector import (
    ObservationCollector,
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


def create_test_dataset() -> Path:
    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix="nexus6_evidence_"
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


def build_evidence_builder(
    dataset_path: Path,
) -> tuple[
    DataEngine,
    DatasetEvidenceBuilder,
]:
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

    builder = DatasetEvidenceBuilder(
        intelligence=intelligence,
        observation_collector=observation_collector,
    )

    return engine, builder


def test_builder_returns_evidence_package() -> None:
    dataset_path = create_test_dataset()

    engine, builder = build_evidence_builder(
        dataset_path
    )

    try:
        package = builder.build()

        assert isinstance(
            package,
            DatasetEvidencePackage,
        )

    finally:
        engine.close()


def test_dataset_metadata_is_preserved() -> None:
    dataset_path = create_test_dataset()

    engine, builder = build_evidence_builder(
        dataset_path
    )

    try:
        package = builder.build()

        assert package.table_name == "dataset"
        assert package.row_count == 12
        assert package.column_count == 5

    finally:
        engine.close()


def test_intelligence_is_included() -> None:
    dataset_path = create_test_dataset()

    engine, builder = build_evidence_builder(
        dataset_path
    )

    try:
        package = builder.build()

        assert package.intelligence

        assert package.intelligence[
            "row_count"
        ] == 12

        assert package.intelligence[
            "column_count"
        ] == 5

        assert "columns" in (
            package.intelligence
        )

        assert "capabilities" in (
            package.intelligence
        )

        assert "potential_targets" in (
            package.intelligence
        )

    finally:
        engine.close()


def test_observations_are_included() -> None:
    dataset_path = create_test_dataset()

    engine, builder = build_evidence_builder(
        dataset_path
    )

    try:
        package = builder.build()

        assert len(
            package.observations
        ) > 0

        assert (
            package.summary.observation_count
            == len(package.observations)
        )

    finally:
        engine.close()


def test_observation_sources_are_summarized() -> None:
    dataset_path = create_test_dataset()

    engine, builder = build_evidence_builder(
        dataset_path
    )

    try:
        package = builder.build()

        sources = (
            package.summary.observation_sources
        )

        assert "relationship_engine" in sources
        assert "statistics_engine" in sources

    finally:
        engine.close()


def test_observation_types_are_summarized() -> None:
    dataset_path = create_test_dataset()

    engine, builder = build_evidence_builder(
        dataset_path
    )

    try:
        package = builder.build()

        types = (
            package.summary.observation_types
        )

        assert "correlation" in types
        assert "distribution" in types
        assert "outliers" in types
        assert "skewness" in types

    finally:
        engine.close()


def test_source_counts_match_observations() -> None:
    dataset_path = create_test_dataset()

    engine, builder = build_evidence_builder(
        dataset_path
    )

    try:
        package = builder.build()

        for source, count in (
            package.summary.source_counts.items()
        ):
            actual_count = len(
                package.get_observations_by_source(
                    source
                )
            )

            assert count == actual_count

    finally:
        engine.close()


def test_type_counts_match_observations() -> None:
    dataset_path = create_test_dataset()

    engine, builder = build_evidence_builder(
        dataset_path
    )

    try:
        package = builder.build()

        for observation_type, count in (
            package.summary.type_counts.items()
        ):
            actual_count = len(
                package.get_observations_by_type(
                    observation_type
                )
            )

            assert count == actual_count

    finally:
        engine.close()


def test_expected_relationship_count() -> None:
    dataset_path = create_test_dataset()

    engine, builder = build_evidence_builder(
        dataset_path
    )

    try:
        package = builder.build()

        relationships = (
            package.get_observations_by_type(
                "correlation"
            )
        )

        assert len(relationships) == 3

    finally:
        engine.close()


def test_package_json_is_valid() -> None:
    dataset_path = create_test_dataset()

    engine, builder = build_evidence_builder(
        dataset_path
    )

    try:
        package = builder.build()

        serialized = package.to_json()

        parsed = json.loads(
            serialized
        )

        assert isinstance(
            parsed,
            dict,
        )

        assert parsed[
            "package_version"
        ] == "1.0"

        assert parsed[
            "table_name"
        ] == "dataset"

        assert parsed[
            "row_count"
        ] == 12

        assert "intelligence" in parsed
        assert "observations" in parsed
        assert "summary" in parsed
        assert "metadata" in parsed

    finally:
        engine.close()


def test_package_round_trip_validation() -> None:
    dataset_path = create_test_dataset()

    engine, builder = build_evidence_builder(
        dataset_path
    )

    try:
        package = builder.build()

        serialized = package.to_json()

        restored = (
            DatasetEvidencePackage.model_validate_json(
                serialized
            )
        )

        assert (
            restored.table_name
            == package.table_name
        )

        assert (
            restored.row_count
            == package.row_count
        )

        assert (
            restored.column_count
            == package.column_count
        )

        assert (
            len(restored.observations)
            == len(package.observations)
        )

        assert (
            restored.summary.observation_count
            == package.summary.observation_count
        )

    finally:
        engine.close()


def test_package_contains_no_interpretation_field() -> None:
    dataset_path = create_test_dataset()

    engine, builder = build_evidence_builder(
        dataset_path
    )

    try:
        package = builder.build()

        package_dict = package.to_dict()

        assert "interpretation" not in package_dict
        assert "recommendation" not in package_dict
        assert "conclusion" not in package_dict

    finally:
        engine.close()


def main() -> None:
    print("=" * 60)
    print("NEXUS-6 DATASET EVIDENCE PACKAGE TEST")
    print("=" * 60)

    test_builder_returns_evidence_package()
    print("PASS: Evidence package creation")

    test_dataset_metadata_is_preserved()
    print("PASS: Dataset metadata")

    test_intelligence_is_included()
    print("PASS: Dataset intelligence")

    test_observations_are_included()
    print("PASS: Observation integration")

    test_observation_sources_are_summarized()
    print("PASS: Observation source summary")

    test_observation_types_are_summarized()
    print("PASS: Observation type summary")

    test_source_counts_match_observations()
    print("PASS: Source count integrity")

    test_type_counts_match_observations()
    print("PASS: Type count integrity")

    test_expected_relationship_count()
    print("PASS: Relationship evidence")

    test_package_json_is_valid()
    print("PASS: JSON serialization")

    test_package_round_trip_validation()
    print("PASS: JSON round-trip validation")

    test_package_contains_no_interpretation_field()
    print("PASS: Evidence/interpretation separation")

    print()
    print("=" * 60)
    print("ALL DATASET EVIDENCE PACKAGE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()