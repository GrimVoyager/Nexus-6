from __future__ import annotations

import json

import pyarrow as pa

from backend.core.loader import DataEngine
from backend.core.quality import DataQualityAnalyzer


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
                "EMP011",
                "EMP012",
            ],
            "Department": [
                "Engineering",
                "Sales",
                "Engineering",
                "Sales",
                "HR",
                "Engineering",
                "HR",
                "Sales",
                "Engineering",
                "HR",
                "Sales",
                "Engineering",
            ],
            "Salary": [
                75000,
                65000,
                85000,
                55000,
                62000,
                90000,
                58000,
                68000,
                82000,
                61000,
                71000,
                88000,
            ],
            "Constant_Value": [
                "ACTIVE",
                "ACTIVE",
                "ACTIVE",
                "ACTIVE",
                "ACTIVE",
                "ACTIVE",
                "ACTIVE",
                "ACTIVE",
                "ACTIVE",
                "ACTIVE",
                "ACTIVE",
                "ACTIVE",
            ],
            "Nullable_Value": [
                "A",
                None,
                "B",
                None,
                "C",
                "D",
                None,
                "E",
                "F",
                None,
                "G",
                "H",
            ],
            "Unique_Value": [
                "U001",
                "U002",
                "U003",
                "U004",
                "U005",
                "U006",
                "U007",
                "U008",
                "U009",
                "U010",
                "U011",
                "U012",
            ],
        }
    )


def find_column(
    report: dict,
    column_name: str,
) -> dict:
    for column in report["columns"]:
        if column["name"] == column_name:
            return column

    raise AssertionError(
        f"Column not found: {column_name}"
    )


def test_report_dimensions(
    report: dict,
) -> None:
    assert report["table_name"] == "dataset"
    assert report["row_count"] == 12
    assert report["column_count"] == 6


def test_missing_values(
    report: dict,
) -> None:
    nullable_column = find_column(
        report,
        "Nullable_Value",
    )

    assert nullable_column["null_count"] == 4
    assert nullable_column["non_null_count"] == 8
    assert nullable_column["null_percentage"] == (
        4 / 12
    ) * 100.0


def test_constant_column(
    report: dict,
) -> None:
    constant_column = find_column(
        report,
        "Constant_Value",
    )

    assert constant_column["distinct_count"] == 1
    assert constant_column["is_constant"] is True


def test_high_cardinality_column(
    report: dict,
) -> None:
    unique_column = find_column(
        report,
        "Unique_Value",
    )

    assert unique_column["distinct_count"] == 12
    assert unique_column["uniqueness_ratio"] == 1.0
    assert unique_column["is_high_cardinality"] is False


def test_numeric_column_detection(
    report: dict,
) -> None:
    salary_column = find_column(
        report,
        "Salary",
    )

    assert salary_column["is_numeric"] is True
    assert salary_column["numeric_validity"]["checked"] is True
    assert (
        salary_column["numeric_validity"][
            "positive_infinity_count"
        ]
        == 0
    )
    assert (
        salary_column["numeric_validity"][
            "negative_infinity_count"
        ]
        == 0
    )
    assert (
        salary_column["numeric_validity"]["nan_count"]
        == 0
    )


def test_duplicate_detection(
    report: dict,
) -> None:
    duplicate_rows = report["duplicate_rows"]

    assert duplicate_rows["duplicate_row_count"] == 0
    assert duplicate_rows["unique_row_count"] == 12
    assert duplicate_rows["duplicate_percentage"] == 0.0


def test_quality_summary(
    report: dict,
) -> None:
    summary = report["summary"]

    assert "Nullable_Value" in (
        summary["columns_with_missing_values"]
    )

    assert "Constant_Value" in (
        summary["constant_columns"]
    )

    assert "has_duplicate_rows" in summary
    assert summary["has_duplicate_rows"] is False

    assert summary["has_quality_issues"] is True


def test_report_is_json_serializable(
    report: dict,
) -> None:
    serialized = json.dumps(
        report,
        default=str,
    )

    parsed = json.loads(serialized)

    assert parsed["row_count"] == 12
    assert parsed["column_count"] == 6


def main() -> None:
    print("=" * 60)
    print("NEXUS-6 DATA QUALITY TEST")
    print("=" * 60)

    table = create_test_table()

    with DataEngine() as engine:
        engine.register_arrow(
            "dataset",
            table,
        )

        analyzer = DataQualityAnalyzer(
            engine,
            table_name="dataset",
        )

        report = analyzer.build_report()

    test_report_dimensions(report)
    print("PASS: Dataset dimensions")

    test_missing_values(report)
    print("PASS: Missing-value analysis")

    test_constant_column(report)
    print("PASS: Constant-column detection")

    test_high_cardinality_column(report)
    print("PASS: High-cardinality analysis")

    test_numeric_column_detection(report)
    print("PASS: Numeric-column validation")

    test_duplicate_detection(report)
    print("PASS: Duplicate-row detection")

    test_quality_summary(report)
    print("PASS: Quality summary")

    test_report_is_json_serializable(report)
    print("PASS: JSON serialization")

    print()
    print("=" * 60)
    print("ALL DATA QUALITY TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
    