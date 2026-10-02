from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pyarrow as pa

from backend.core.loader import DataEngine
from backend.core.planner import create_execution_plan
from backend.core.sanitizer import DataSanitizer


def create_fake_csv(file_path: Path) -> None:
    """
    Create a small CSV dataset containing representative PII.
    """

    rows = [
        {
            "Employee_Name": "Anant Chaurasia",
            "Email": "anant@example.com",
            "Phone": "9876543210",
            "PAN": "ABCDE1234F",
            "Salary": 75000,
        },
        {
            "Employee_Name": "Rahul Sharma",
            "Email": "rahul@example.com",
            "Phone": "9123456789",
            "PAN": "FGHIJ5678K",
            "Salary": 65000,
        },
        {
            "Employee_Name": "Priya Singh",
            "Email": "priya@example.com",
            "Phone": "9988776655",
            "PAN": "LMNOP1234Q",
            "Salary": 85000,
        },
    ]

    with file_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(file_path: Path) -> list[dict[str, str]]:
    """
    Read CSV rows using Python's standard library.
    """

    with file_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def sanitize_rows(
    rows: list[dict[str, str]],
    sanitizer: DataSanitizer,
) -> list[dict[str, str]]:
    """
    Sanitize every string value in the dataset.
    """

    sanitized_rows: list[dict[str, str]] = []

    for row in rows:
        sanitized_row: dict[str, str] = {}

        for column, value in row.items():
            sanitized_row[column] = sanitizer.sanitize_text(value)

        sanitized_rows.append(sanitized_row)

    return sanitized_rows


def rows_to_arrow(
    rows: list[dict[str, str]],
) -> pa.Table:
    """
    Convert sanitized Python rows into a PyArrow table.

    Numeric salary values are converted back to integers so DuckDB can
    perform numerical aggregation on the column.
    """

    names = [row["Employee_Name"] for row in rows]
    emails = [row["Email"] for row in rows]
    phones = [row["Phone"] for row in rows]
    pans = [row["PAN"] for row in rows]
    salaries = [int(row["Salary"]) for row in rows]

    return pa.table(
        {
            "Employee_Name": names,
            "Email": emails,
            "Phone": phones,
            "PAN": pans,
            "Salary": salaries,
        }
    )


def test_sanitization(
    rows: list[dict[str, str]],
) -> None:
    """
    Verify that expected PII patterns were replaced.
    """

    for row in rows:
        assert "<EMAIL_ADDRESS>" in row["Email"]
        assert "<PHONE_NUMBER>" in row["Phone"]
        assert "<IN_PAN>" in row["PAN"]

    print("PASS: PII sanitization")


def test_duckdb(
    table: pa.Table,
) -> None:
    """
    Verify Arrow registration and analytical SQL execution.
    """

    with DataEngine() as engine:
        engine.register_arrow(
            "employees",
            table,
        )

        result = engine.fetch_dicts(
            """
            SELECT
                COUNT(*) AS employee_count,
                AVG(Salary) AS average_salary,
                MAX(Salary) AS maximum_salary
            FROM employees
            """
        )

        assert len(result) == 1

        row = result[0]

        assert row["employee_count"] == 3
        assert row["average_salary"] == 75000.0
        assert row["maximum_salary"] == 85000

        metadata = engine.inspect_metadata(
            "employees"
        )

        assert metadata["row_count"] == 3
        assert metadata["column_count"] == 5

    print("PASS: DuckDB analytical execution")
    print("PASS: Dataset metadata inspection")


def test_execution_plan() -> None:
    """
    Verify that the structured execution plan is valid.
    """

    plan = create_execution_plan(
        detected_domain="hr_workforce",
        overlay=True,
        primary_sql=(
            "SELECT Employee_Name, Salary "
            "FROM employees "
            "ORDER BY Salary DESC"
        ),
        target="Salary",
        reasoning=(
            "The dataset represents workforce information "
            "and should use the HR domain overlay."
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

    assert plan.detected_domain == "hr_workforce"
    assert plan.overlay is True
    assert plan.target == "Salary"

    schema = plan.model_json_schema()

    assert schema["additionalProperties"] is False
    assert "selected_tiers" in schema["properties"]
    assert "detected_domain" in schema["properties"]

    serialized = plan.model_dump_json()

    parsed = json.loads(serialized)

    assert parsed["detected_domain"] == "hr_workforce"
    assert parsed["selected_tiers"] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]

    print("PASS: ExecutionPlan validation")
    print("PASS: ExecutionPlan JSON serialization")
    print("PASS: ExecutionPlan JSON schema")


def main() -> None:
    """
    Run all Phase 1 integration tests.
    """

    print("=" * 60)
    print("NEXUS-6 PHASE 1 INTEGRATION TEST")
    print("=" * 60)

    sanitizer = DataSanitizer()

    with tempfile.TemporaryDirectory(
        prefix="nexus6_phase1_"
    ) as temporary_directory:

        csv_path = Path(
            temporary_directory
        ) / "employees.csv"

        create_fake_csv(csv_path)

        print()
        print(f"Test dataset: {csv_path}")

        rows = read_csv_rows(csv_path)

        print(
            f"Rows loaded: {len(rows)}"
        )

        sanitized_rows = sanitize_rows(
            rows,
            sanitizer,
        )

        test_sanitization(
            sanitized_rows
        )

        arrow_table = rows_to_arrow(
            sanitized_rows
        )

        test_duckdb(
            arrow_table
        )

        test_execution_plan()

    print()
    print("=" * 60)
    print("ALL PHASE 1 TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
    