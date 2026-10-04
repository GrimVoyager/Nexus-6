from __future__ import annotations

import json

import pyarrow as pa
import pytest

from backend.core.intelligence import DatasetIntelligence
from backend.core.loader import DataEngine



def create_test_table() -> pa.Table:
    """
    Create a deterministic dataset for intelligence testing.
    """

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
            "Employee_Name": [
                "Anant",
                "Rahul",
                "Priya",
                "Aman",
                "Neha",
                "Rohan",
                "Karan",
                "Simran",
                "Vikas",
                "Aditi",
                "Arjun",
                "Meera",
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
            "Experience": [
                2,
                4,
                5,
                1,
                3,
                7,
                2,
                4,
                6,
                3,
                5,
                8,
            ],
            "Performance": [
                82,
                76,
                91,
                70,
                74,
                95,
                79,
                81,
                89,
                77,
                84,
                93,
            ],
        }
    )

@pytest.fixture
def profile() -> dict:
    table = create_test_table()

    with DataEngine() as engine:
        engine.register_arrow(
            "dataset",
            table,
        )

        intelligence = DatasetIntelligence(
            engine,
            table_name="dataset",
        )

        return intelligence.build_profile()


def find_column(
    profile: dict,
    column_name: str,
) -> dict:
    """
    Find a column profile by name.
    """

    for column in profile["columns"]:
        if column["name"] == column_name:
            return column

    raise AssertionError(
        f"Column not found: {column_name}"
    )


def test_dataset_dimensions(
    profile: dict,
) -> None:
    """
    Verify basic dataset dimensions.
    """

    assert profile["table_name"] == "dataset"
    assert profile["row_count"] == 12
    assert profile["column_count"] == 6


def test_semantic_types(
    profile: dict,
) -> None:
    """
    Verify deterministic semantic classification.
    """

    employee_id = find_column(
        profile,
        "Employee_ID",
    )

    employee_name = find_column(
        profile,
        "Employee_Name",
    )

    department = find_column(
        profile,
        "Department",
    )

    salary = find_column(
        profile,
        "Salary",
    )

    experience = find_column(
        profile,
        "Experience",
    )

    performance = find_column(
        profile,
        "Performance",
    )

    assert employee_id["semantic_type"] == "text"
    assert employee_name["semantic_type"] == "text"
    assert department["semantic_type"] == "categorical"
    assert salary["semantic_type"] == "numeric"
    assert experience["semantic_type"] == "numeric"
    assert performance["semantic_type"] == "numeric"


def test_numeric_statistics(
    profile: dict,
) -> None:
    """
    Verify that numeric statistics are calculated correctly.
    """

    statistics = {
        item["column"]: item
        for item in profile["numeric_statistics"]
    }

    assert set(statistics.keys()) == {
        "Salary",
        "Experience",
        "Performance",
    }

    assert statistics["Salary"]["minimum"] == 55000
    assert statistics["Salary"]["maximum"] == 90000
    assert abs(
        statistics["Salary"]["mean"] - 71666.66666666667
    ) < 0.000001

    assert statistics["Experience"]["minimum"] == 1
    assert statistics["Experience"]["maximum"] == 8

    assert statistics["Performance"]["minimum"] == 70
    assert statistics["Performance"]["maximum"] == 95


def test_potential_targets(
    profile: dict,
) -> None:
    """
    Verify that identifier and name columns are excluded from
    potential target candidates.
    """

    target_names = {
        item["column"]
        for item in profile["potential_targets"]
    }

    assert "Employee_ID" not in target_names
    assert "Employee_Name" not in target_names

    assert "Department" in target_names
    assert "Salary" in target_names
    assert "Experience" in target_names
    assert "Performance" in target_names


def test_capabilities(
    profile: dict,
) -> None:
    """
    Verify analytical capability detection.
    """

    capabilities = profile["capabilities"]

    assert capabilities["query"] is True
    assert capabilities["overview"] is True
    assert capabilities["audit"] is True
    assert capabilities["repair"] is True
    assert capabilities["relationships"] is True
    assert capabilities["categorical_analysis"] is True
    assert capabilities["time_series_analysis"] is False
    assert capabilities["predictive_modeling"] is True
    assert capabilities["engineering_analysis"] is True


def test_profile_is_json_serializable(
    profile: dict,
) -> None:
    """
    Verify that the intelligence profile can be serialized for API,
    storage, or LLM consumption.
    """

    serialized = json.dumps(
        profile,
        default=str,
    )

    parsed = json.loads(serialized)

    assert parsed["row_count"] == 12
    assert parsed["column_count"] == 6


def main() -> None:
    """
    Execute the Dataset Intelligence test suite.
    """

    print("=" * 60)
    print("NEXUS-6 DATASET INTELLIGENCE TEST")
    print("=" * 60)

    table = create_test_table()

    with DataEngine() as engine:
        engine.register_arrow(
            "dataset",
            table,
        )

        intelligence = DatasetIntelligence(
            engine,
            table_name="dataset",
        )

        profile = intelligence.build_profile()

    test_dataset_dimensions(profile)
    print("PASS: Dataset dimensions")

    test_semantic_types(profile)
    print("PASS: Semantic type classification")

    test_numeric_statistics(profile)
    print("PASS: Numeric statistics")

    test_potential_targets(profile)
    print("PASS: Potential target detection")

    test_capabilities(profile)
    print("PASS: Analytical capability detection")

    test_profile_is_json_serializable(profile)
    print("PASS: JSON serialization")

    print()
    print("=" * 60)
    print("ALL DATASET INTELLIGENCE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()