from __future__ import annotations

import json

import pyarrow as pa

from backend.core.loader import DataEngine
from backend.core.statistics import StatisticalAnalyzer
import pytest

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
            ],
            "Salary": [
                50000,
                52000,
                54000,
                56000,
                58000,
                60000,
                62000,
                64000,
                66000,
                68000,
                150000,
            ],
            "Experience": [
                1,
                2,
                2,
                3,
                3,
                4,
                5,
                5,
                6,
                7,
                8,
            ],
            "Signed_Value": [
                -5,
                -2,
                0,
                1,
                3,
                4,
                6,
                8,
                10,
                12,
                15,
            ],
        }
    )

@pytest.fixture
def report() -> dict:
    table = create_test_table()

    with DataEngine() as engine:
        engine.register_arrow(
            "dataset",
            table,
        )

        analyzer = StatisticalAnalyzer(
            engine,
            table_name="dataset",
        )

        return analyzer.build_report()

def find_statistics(
    report: dict,
    column_name: str,
) -> dict:
    for item in report["statistics"]:
        if item["column"] == column_name:
            return item

    raise AssertionError(
        f"Statistics not found: {column_name}"
    )


def test_numeric_columns(
    report: dict,
) -> None:
    assert report["numeric_column_count"] == 3

    assert set(report["numeric_columns"]) == {
        "Salary",
        "Experience",
        "Signed_Value",
    }


def test_salary_statistics(
    report: dict,
) -> None:
    salary = find_statistics(
        report,
        "Salary",
    )

    assert salary["sample_size"] == 11
    assert salary["minimum"] == 50000.0
    assert salary["maximum"] == 150000.0

    assert salary["median"] == 60000.0

    assert salary["quartiles"]["q2"] == 60000.0

    assert salary["percentiles"]["p50"] == 60000.0

    assert salary["outlier_count"] == 1
    assert salary["status"] == "success"


def test_experience_statistics(
    report: dict,
) -> None:
    experience = find_statistics(
        report,
        "Experience",
    )

    assert experience["sample_size"] == 11
    assert experience["minimum"] == 1.0
    assert experience["maximum"] == 8.0
    assert experience["median"] == 4.0

    assert experience["zero_count"] == 0
    assert experience["negative_count"] == 0


def test_negative_and_zero_counts(
    report: dict,
) -> None:
    signed_value = find_statistics(
        report,
        "Signed_Value",
    )

    assert signed_value["zero_count"] == 1
    assert signed_value["negative_count"] == 2


def test_skewness_is_calculated(
    report: dict,
) -> None:
    salary = find_statistics(
        report,
        "Salary",
    )

    assert salary["skewness"] is not None
    assert salary["skewness"] > 0


def test_outlier_bounds(
    report: dict,
) -> None:
    salary = find_statistics(
        report,
        "Salary",
    )

    bounds = salary["outlier_bounds"]

    assert bounds["lower"] < 50000
    assert bounds["upper"] < 150000


def test_report_is_json_serializable(
    report: dict,
) -> None:
    serialized = json.dumps(
        report,
        default=str,
    )

    parsed = json.loads(serialized)

    assert parsed["numeric_column_count"] == 3
    assert len(parsed["statistics"]) == 3


def main() -> None:
    print("=" * 60)
    print("NEXUS-6 STATISTICAL INTELLIGENCE TEST")
    print("=" * 60)

    table = create_test_table()

    with DataEngine() as engine:
        engine.register_arrow(
            "dataset",
            table,
        )

        analyzer = StatisticalAnalyzer(
            engine,
            table_name="dataset",
        )

        report = analyzer.build_report()

    test_numeric_columns(report)
    print("PASS: Numeric-column detection")

    test_salary_statistics(report)
    print("PASS: Descriptive statistics")

    test_experience_statistics(report)
    print("PASS: Median and range statistics")

    test_negative_and_zero_counts(report)
    print("PASS: Zero and negative-value analysis")

    test_skewness_is_calculated(report)
    print("PASS: Skewness calculation")

    test_outlier_bounds(report)
    print("PASS: IQR outlier detection")

    test_report_is_json_serializable(report)
    print("PASS: JSON serialization")

    print()
    print("=" * 60)
    print("ALL STATISTICAL INTELLIGENCE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()