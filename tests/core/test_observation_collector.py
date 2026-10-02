from __future__ import annotations

import json
import tempfile
from pathlib import Path

from backend.core.loader import DataEngine
from backend.core.observation_collector import (
    ObservationCollector,
)
from backend.core.quality import DataQualityAnalyzer
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
            prefix="nexus6_observation_"
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


def build_collector(
    dataset_path: Path,
) -> tuple[
    DataEngine,
    ObservationCollector,
]:
    engine = DataEngine()

    engine.read_file(
        dataset_path,
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

    return engine, collector


def test_collector_returns_store() -> None:
    dataset_path = create_test_dataset()

    engine, collector = build_collector(
        dataset_path
    )

    try:
        store = collector.collect()

        assert store.count() > 0
    finally:
        engine.close()


def test_relationship_observations_are_collected() -> None:
    dataset_path = create_test_dataset()

    engine, collector = build_collector(
        dataset_path
    )

    try:
        store = collector.collect()

        observations = store.filter(
            observation_type="correlation"
        )

        assert len(observations) > 0

        for observation in observations:
            assert observation.source == (
                "relationship_engine"
            )

            assert len(
                observation.variables
            ) == 2

            assert observation.sample_size == 12

            assert "pearson" in (
                observation.statistic
            )

            assert "spearman" in (
                observation.statistic
            )
    finally:
        engine.close()


def test_statistical_observations_are_collected() -> None:
    dataset_path = create_test_dataset()

    engine, collector = build_collector(
        dataset_path
    )

    try:
        store = collector.collect()

        distributions = store.filter(
            observation_type="distribution"
        )

        outliers = store.filter(
            observation_type="outliers"
        )

        skewness = store.filter(
            observation_type="skewness"
        )

        assert len(distributions) == 3
        assert len(outliers) == 3
        assert len(skewness) == 3

        for observation in distributions:
            assert observation.source == (
                "statistics_engine"
            )

            assert observation.sample_size == 12
            assert observation.method == (
                "descriptive_statistics"
            )

    finally:
        engine.close()


def test_quality_observations_are_supported() -> None:
    dataset_path = create_test_dataset()

    engine, collector = build_collector(
        dataset_path
    )

    try:
        store = collector.collect()

        quality_observations = store.filter(
            source="quality_engine"
        )

        assert isinstance(
            quality_observations,
            list,
        )
    finally:
        engine.close()


def test_all_observations_have_required_identity() -> None:
    dataset_path = create_test_dataset()

    engine, collector = build_collector(
        dataset_path
    )

    try:
        store = collector.collect()

        assert store.count() > 0

        for observation in store.all():
            assert observation.observation_id
            assert observation.type
            assert observation.source

            assert observation.observation_id.startswith(
                "obs_"
            )
    finally:
        engine.close()


def test_observation_sources_are_separated() -> None:
    quality_csv_content = """Employee_Name,Department,Experience,Salary,Performance
Alice,Engineering,2,50000,72
Bob,Engineering,3,,75
Charlie,HR,5,60000,81
Charlie,HR,5,60000,81
"""

    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix="nexus6_quality_observation_"
        )
    )

    quality_dataset_path = (
        temporary_directory
        / "quality_test.csv"
    )

    quality_dataset_path.write_text(
        quality_csv_content,
        encoding="utf-8",
    )

    engine, collector = build_collector(
        quality_dataset_path
    )

    try:
        store = collector.collect()

        sources = {
            observation.source
            for observation in store.all()
        }

        assert "relationship_engine" in sources
        assert "statistics_engine" in sources
        assert "quality_engine" in sources

        quality_observations = store.filter(
            source="quality_engine"
        )

        assert len(quality_observations) > 0

        quality_types = {
            observation.type
            for observation in quality_observations
        }

        assert "missing_values" in quality_types
        assert "duplicate_rows" in quality_types

    finally:
        engine.close()


def test_observation_store_json_is_valid() -> None:
    dataset_path = create_test_dataset()

    engine, collector = build_collector(
        dataset_path
    )

    try:
        store = collector.collect()

        serialized = store.to_json()

        parsed = json.loads(
            serialized
        )

        assert isinstance(parsed, list)
        assert len(parsed) == store.count()

        for item in parsed:
            assert "observation_id" in item
            assert "type" in item
            assert "source" in item
    finally:
        engine.close()


def test_expected_relationship_count() -> None:
    dataset_path = create_test_dataset()

    engine, collector = build_collector(
        dataset_path
    )

    try:
        store = collector.collect()

        relationships = store.filter(
            observation_type="correlation"
        )

        # The dataset contains exactly three numeric
        # columns: Experience, Salary, Performance.
        # Three numeric columns create three unique pairs.
        assert len(relationships) == 3
    finally:
        engine.close()


def main() -> None:
    print("=" * 60)
    print("NEXUS-6 OBSERVATION COLLECTOR TEST")
    print("=" * 60)

    test_collector_returns_store()
    print("PASS: Collector returns ObservationStore")

    test_relationship_observations_are_collected()
    print("PASS: Relationship observations")

    test_statistical_observations_are_collected()
    print("PASS: Statistical observations")

    test_quality_observations_are_supported()
    print("PASS: Quality observations")

    test_all_observations_have_required_identity()
    print("PASS: Observation identity")

    test_observation_sources_are_separated()
    print("PASS: Source separation")

    test_observation_store_json_is_valid()
    print("PASS: JSON evidence package")

    test_expected_relationship_count()
    print("PASS: Expected relationship count")

    print()
    print("=" * 60)
    print("ALL OBSERVATION COLLECTOR TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()