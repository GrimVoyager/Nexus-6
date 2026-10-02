from __future__ import annotations

from backend.core.execution_observation_adapter import (
    ExecutionObservationAdapter,
)
from backend.core.executor import (
    TierExecutionResult,
)
from backend.core.planner import (
    create_execution_plan,
)


def create_execution_result(
    *,
    executed_tiers: list[int],
    partial_tiers: list[int],
    skipped_tiers: list[int],
    execution_details: dict | None = None,
) -> TierExecutionResult:
    plan = create_execution_plan(
        detected_domain="general",
        overlay=False,
        primary_sql=None,
        target=None,
        reasoning="Adapter test plan.",
    )

    return TierExecutionResult(
        plan=plan,
        executed_tiers=executed_tiers,
        partial_tiers=partial_tiers,
        skipped_tiers=skipped_tiers,
        observations=None,
        execution_details=(
            execution_details
            or {}
        ),
    )


def test_executed_tier_creates_observation() -> None:
    result = create_execution_result(
        executed_tiers=[2],
        partial_tiers=[],
        skipped_tiers=[1, 3, 4, 5, 6],
    )

    store = (
        ExecutionObservationAdapter()
        .collect(result)
    )

    assert store.count() == 1

    observation = store.all()[0]

    assert observation.type == "tier_execution"
    assert observation.variables == ["tier_2"]
    assert (
        observation.statistic[
            "execution_status"
        ]
        == "EXECUTED"
    )


def test_partial_tier_creates_observation() -> None:
    result = create_execution_result(
        executed_tiers=[2],
        partial_tiers=[5],
        skipped_tiers=[1, 3, 4, 6],
    )

    store = (
        ExecutionObservationAdapter()
        .collect(result)
    )

    assert store.count() == 2

    observations = store.all()

    statuses = {
        observation.statistic[
            "execution_status"
        ]
        for observation in observations
    }

    assert statuses == {
        "EXECUTED",
        "PARTIAL",
    }


def test_skipped_tiers_do_not_create_observations() -> None:
    result = create_execution_result(
        executed_tiers=[],
        partial_tiers=[],
        skipped_tiers=[1, 2, 3, 4, 5, 6],
    )

    store = (
        ExecutionObservationAdapter()
        .collect(result)
    )

    assert store.count() == 0


def test_execution_details_are_preserved() -> None:
    details = {
        "2": {
            "status": "completed",
            "rows_analyzed": 100,
        }
    }

    result = create_execution_result(
        executed_tiers=[2],
        partial_tiers=[],
        skipped_tiers=[1, 3, 4, 5, 6],
        execution_details=details,
    )

    store = (
        ExecutionObservationAdapter()
        .collect(result)
    )

    observation = store.all()[0]

    assert (
        observation.statistic["details"]
        == details["2"]
    )


def test_observation_source_is_correct() -> None:
    result = create_execution_result(
        executed_tiers=[3],
        partial_tiers=[],
        skipped_tiers=[1, 2, 4, 5, 6],
    )

    store = (
        ExecutionObservationAdapter()
        .collect(result)
    )

    observation = store.all()[0]

    assert (
        observation.source
        == "execution_observation_adapter"
    )


def test_observation_method_is_deterministic() -> None:
    result = create_execution_result(
        executed_tiers=[5],
        partial_tiers=[],
        skipped_tiers=[1, 2, 3, 4, 6],
    )

    store = (
        ExecutionObservationAdapter()
        .collect(result)
    )

    observation = store.all()[0]

    assert (
        observation.method
        == "deterministic_execution_adapter"
    )


def test_same_execution_produces_same_observation_id() -> None:
    result_one = create_execution_result(
        executed_tiers=[2, 3],
        partial_tiers=[],
        skipped_tiers=[1, 4, 5, 6],
    )

    result_two = create_execution_result(
        executed_tiers=[2, 3],
        partial_tiers=[],
        skipped_tiers=[1, 4, 5, 6],
    )

    adapter = ExecutionObservationAdapter()

    store_one = adapter.collect(
        result_one
    )

    store_two = adapter.collect(
        result_two
    )

    ids_one = [
        observation.observation_id
        for observation in store_one.all()
    ]

    ids_two = [
        observation.observation_id
        for observation in store_two.all()
    ]

    assert ids_one == ids_two


def test_multiple_executed_tiers_create_multiple_observations() -> None:
    result = create_execution_result(
        executed_tiers=[2, 3, 5],
        partial_tiers=[],
        skipped_tiers=[1, 4, 6],
    )

    store = (
        ExecutionObservationAdapter()
        .collect(result)
    )

    assert store.count() == 3

    variables = {
        observation.variables[0]
        for observation in store.all()
    }

    assert variables == {
        "tier_2",
        "tier_3",
        "tier_5",
    }


def main() -> None:
    print("=" * 60)
    print("NEXUS-6 EXECUTION OBSERVATION ADAPTER TEST")
    print("=" * 60)

    test_executed_tier_creates_observation()
    print("PASS: Executed tier creates observation")

    test_partial_tier_creates_observation()
    print("PASS: Partial tier creates observation")

    test_skipped_tiers_do_not_create_observations()
    print("PASS: Skipped tiers excluded")

    test_execution_details_are_preserved()
    print("PASS: Execution details preserved")

    test_observation_source_is_correct()
    print("PASS: Observation source validated")

    test_observation_method_is_deterministic()
    print("PASS: Deterministic method validated")

    test_same_execution_produces_same_observation_id()
    print("PASS: Deterministic observation IDs")

    test_multiple_executed_tiers_create_multiple_observations()
    print("PASS: Multiple tier observations")

    print()
    print("=" * 60)
    print(
        "ALL NEXUS-6 EXECUTION OBSERVATION ADAPTER "
        "TESTS PASSED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()