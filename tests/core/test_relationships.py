from __future__ import annotations

import json

import pyarrow as pa

from backend.core.loader import DataEngine
from backend.core.relationships import RelationshipAnalyzer


def create_test_table() -> pa.Table:
    return pa.table(
        {
            "Employee_ID": [
                "EMP001",
                "EMP002",
                "EMP003",
                "EMP004",
                "EMP005",
                "EMP006",
                "EMP007",
                "EMP008",
                "EMP009",
                "EMP010",
            ],
            "Experience": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
            "Salary": [
                50000,
                54000,
                58000,
                62000,
                66000,
                70000,
                74000,
                78000,
                82000,
                86000,
            ],
            "Performance": [
                60,
                65,
                68,
                72,
                75,
                78,
                80,
                84,
                87,
                90,
            ],
            "Constant_Value": [
                100,
                100,
                100,
                100,
                100,
                100,
                100,
                100,
                100,
                100,
            ],
        }
    )


def find_relationship(
    report: dict,
    first_column: str,
    second_column: str,
) -> dict:
    expected_variables = {
        first_column,
        second_column,
    }

    for relationship in report["relationships"]:
        if (
            set(relationship["variables"])
            == expected_variables
        ):
            return relationship

    raise AssertionError(
        "Relationship not found: "
        f"{first_column} <-> {second_column}"
    )


def test_numeric_columns(
    report: dict,
) -> None:
    assert report["numeric_column_count"] == 4

    assert set(report["numeric_columns"]) == {
        "Experience",
        "Salary",
        "Performance",
        "Constant_Value",
    }


def test_relationship_count(
    report: dict,
) -> None:
    assert report["relationship_count"] == 6


def test_strong_positive_relationship(
    report: dict,
) -> None:
    relationship = find_relationship(
        report,
        "Experience",
        "Salary",
    )

    pearson = relationship[
        "statistic"
    ]["pearson"]

    spearman = relationship[
        "statistic"
    ]["spearman"]

    assert relationship["type"] == "correlation"
    assert relationship["sample_size"] == 10

    assert pearson["value"] > 0.99
    assert pearson["direction"] == "positive"
    assert pearson["strength"] == "strong"

    assert spearman["value"] > 0.99
    assert spearman["direction"] == "positive"
    assert spearman["strength"] == "strong"


def test_second_strong_relationship(
    report: dict,
) -> None:
    relationship = find_relationship(
        report,
        "Experience",
        "Performance",
    )

    pearson = relationship[
        "statistic"
    ]["pearson"]

    assert pearson["value"] > 0.99
    assert pearson["direction"] == "positive"
    assert pearson["strength"] == "strong"


def test_constant_column_handling(
    report: dict,
) -> None:
    relationship = find_relationship(
        report,
        "Experience",
        "Constant_Value",
    )

    assert (
        relationship["status"]
        == "insufficient_variation"
    )

    assert relationship["pearson"] is None
    assert relationship["spearman"] is None


def test_observation_source(
    report: dict,
) -> None:
    relationship = find_relationship(
        report,
        "Experience",
        "Salary",
    )

    assert (
        relationship["source"]
        == "relationship_engine"
    )

    assert relationship["method"] == [
        "pearson",
        "spearman",
    ]


def test_report_is_json_serializable(
    report: dict,
) -> None:
    serialized = json.dumps(
        report,
        default=str,
    )

    parsed = json.loads(serialized)

    assert parsed["numeric_column_count"] == 4
    assert parsed["relationship_count"] == 6


def main() -> None:
    print("=" * 60)
    print("NEXUS-6 RELATIONSHIP ENGINE TEST")
    print("=" * 60)

    table = create_test_table()

    with DataEngine() as engine:
        engine.register_arrow(
            "dataset",
            table,
        )

        analyzer = RelationshipAnalyzer(
            engine,
            table_name="dataset",
        )

        report = analyzer.build_report()

    test_numeric_columns(report)
    print("PASS: Numeric-column detection")

    test_relationship_count(report)
    print("PASS: Pairwise relationship generation")

    test_strong_positive_relationship(report)
    print("PASS: Pearson/Spearman relationship analysis")

    test_second_strong_relationship(report)
    print("PASS: Multiple relationship analysis")

    test_constant_column_handling(report)
    print("PASS: Constant-column handling")

    test_observation_source(report)
    print("PASS: Observation metadata")

    test_report_is_json_serializable(report)
    print("PASS: JSON serialization")

    print()
    print("=" * 60)
    print("ALL RELATIONSHIP ENGINE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()