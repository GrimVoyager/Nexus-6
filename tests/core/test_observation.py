from __future__ import annotations

import json

from backend.core.observation import (
    Observation,
    ObservationStore,
    create_observation_id,
)


def create_test_observations() -> list[Observation]:
    correlation_id = create_observation_id(
        observation_type="correlation",
        variables=[
            "Experience",
            "Salary",
        ],
        source="relationship_engine",
        method="pearson",
    )

    missing_id = create_observation_id(
        observation_type="missing_values",
        variables=[
            "Salary",
        ],
        source="quality_engine",
        method="null_count",
    )

    distribution_id = create_observation_id(
        observation_type="distribution",
        variables=[
            "Salary",
        ],
        source="statistics_engine",
        method="descriptive_statistics",
    )

    return [
        Observation(
            observation_id=correlation_id,
            type="correlation",
            variables=[
                "Experience",
                "Salary",
            ],
            statistic={
                "name": "pearson_r",
                "value": 0.81,
            },
            sample_size=100,
            method="pearson",
            source="relationship_engine",
        ),
        Observation(
            observation_id=missing_id,
            type="missing_values",
            variables=[
                "Salary",
            ],
            statistic={
                "name": "null_count",
                "value": 4,
                "percentage": 4.0,
            },
            sample_size=100,
            method="null_count",
            source="quality_engine",
        ),
        Observation(
            observation_id=distribution_id,
            type="distribution",
            variables=[
                "Salary",
            ],
            statistic={
                "mean": 70000.0,
                "median": 65000.0,
                "iqr": 15000.0,
            },
            sample_size=96,
            method="descriptive_statistics",
            source="statistics_engine",
        ),
    ]


def test_observation_model() -> None:
    observation = Observation(
        observation_id="obs_test_001",
        type="Correlation",
        variables=[
            "Experience",
            "Salary",
        ],
        statistic={
            "pearson_r": 0.81,
        },
        sample_size=100,
        method="pearson",
        source="Relationship_Engine",
    )

    assert observation.observation_id == "obs_test_001"
    assert observation.type == "correlation"
    assert observation.source == "relationship_engine"
    assert observation.variables == [
        "Experience",
        "Salary",
    ]
    assert observation.sample_size == 100


def test_deterministic_observation_id() -> None:
    first_id = create_observation_id(
        observation_type="correlation",
        variables=[
            "Experience",
            "Salary",
        ],
        source="relationship_engine",
        method="pearson",
    )

    second_id = create_observation_id(
        observation_type="correlation",
        variables=[
            "Experience",
            "Salary",
        ],
        source="relationship_engine",
        method="pearson",
    )

    assert first_id == second_id
    assert first_id.startswith("obs_")


def test_different_observation_identity_creates_different_id() -> None:
    first_id = create_observation_id(
        observation_type="correlation",
        variables=[
            "Experience",
            "Salary",
        ],
        source="relationship_engine",
        method="pearson",
    )

    second_id = create_observation_id(
        observation_type="correlation",
        variables=[
            "Experience",
            "Performance",
        ],
        source="relationship_engine",
        method="pearson",
    )

    assert first_id != second_id


def test_store_add_and_get() -> None:
    observations = create_test_observations()

    store = ObservationStore()

    stored = store.add(
        observations[0]
    )

    assert stored.observation_id == (
        observations[0].observation_id
    )

    retrieved = store.get(
        observations[0].observation_id
    )

    assert retrieved is not None
    assert retrieved.to_dict() == (
        observations[0].to_dict()
    )

    assert store.count() == 1


def test_store_add_many() -> None:
    observations = create_test_observations()

    store = ObservationStore()

    stored = store.add_many(
        observations
    )

    assert len(stored) == 3
    assert store.count() == 3


def test_store_filters() -> None:
    observations = create_test_observations()

    store = ObservationStore()

    store.add_many(
        observations
    )

    correlations = store.filter(
        observation_type="CORRELATION"
    )

    assert len(correlations) == 1
    assert correlations[0].type == "correlation"

    quality = store.filter(
        source="QUALITY_ENGINE"
    )

    assert len(quality) == 1
    assert quality[0].source == "quality_engine"

    statistics = store.filter(
        observation_type="distribution",
        source="statistics_engine",
    )

    assert len(statistics) == 1
    assert statistics[0].type == "distribution"


def test_store_is_idempotent() -> None:
    observations = create_test_observations()

    store = ObservationStore()

    first = store.add(
        observations[0]
    )

    second = store.add(
        observations[0]
    )

    assert first is second
    assert store.count() == 1


def test_store_rejects_conflicting_duplicate_id() -> None:
    observation = Observation(
        observation_id="obs_conflict",
        type="correlation",
        variables=[
            "Experience",
            "Salary",
        ],
        statistic={
            "value": 0.81,
        },
        sample_size=100,
        method="pearson",
        source="relationship_engine",
    )

    conflicting = Observation(
        observation_id="obs_conflict",
        type="correlation",
        variables=[
            "Experience",
            "Salary",
        ],
        statistic={
            "value": 0.91,
        },
        sample_size=100,
        method="pearson",
        source="relationship_engine",
    )

    store = ObservationStore()

    store.add(observation)

    try:
        store.add(conflicting)
    except ValueError as error:
        assert "already exists" in str(error)
    else:
        raise AssertionError(
            "Expected ValueError for conflicting observation ID"
        )


def test_require_raises_for_missing_observation() -> None:
    store = ObservationStore()

    try:
        store.require("obs_missing")
    except KeyError as error:
        assert "obs_missing" in str(error)
    else:
        raise AssertionError(
            "Expected KeyError for missing observation"
        )


def test_remove_and_clear() -> None:
    observations = create_test_observations()

    store = ObservationStore()

    store.add_many(
        observations
    )

    removed = store.remove(
        observations[0].observation_id
    )

    assert removed is True
    assert store.count() == 2

    removed_again = store.remove(
        observations[0].observation_id
    )

    assert removed_again is False

    store.clear()

    assert store.count() == 0
    assert store.all() == []


def test_json_serialization() -> None:
    observations = create_test_observations()

    store = ObservationStore()

    store.add_many(
        observations
    )

    observation_json = observations[0].to_json()

    parsed_observation = json.loads(
        observation_json
    )

    assert (
        parsed_observation["type"]
        == "correlation"
    )

    store_json = store.to_json()

    parsed_store = json.loads(
        store_json
    )

    assert isinstance(parsed_store, list)
    assert len(parsed_store) == 3


def main() -> None:
    print("=" * 60)
    print("NEXUS-6 OBSERVATION STORE TEST")
    print("=" * 60)

    test_observation_model()
    print("PASS: Observation model")

    test_deterministic_observation_id()
    print("PASS: Deterministic observation IDs")

    test_different_observation_identity_creates_different_id()
    print("PASS: Observation identity separation")

    test_store_add_and_get()
    print("PASS: Add and retrieve")

    test_store_add_many()
    print("PASS: Batch insertion")

    test_store_filters()
    print("PASS: Type/source filtering")

    test_store_is_idempotent()
    print("PASS: Idempotent insertion")

    test_store_rejects_conflicting_duplicate_id()
    print("PASS: Conflicting ID protection")

    test_require_raises_for_missing_observation()
    print("PASS: Required observation lookup")

    test_remove_and_clear()
    print("PASS: Remove and clear")

    test_json_serialization()
    print("PASS: JSON serialization")

    print()
    print("=" * 60)
    print("ALL OBSERVATION STORE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()