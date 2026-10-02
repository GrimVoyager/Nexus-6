from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Observation(BaseModel):
    """
    Standard evidence unit used by the NEXUS-6 analytical system.

    An observation records a factual analytical result produced by a
    deterministic engine. It does not contain an interpretation,
    recommendation, or LLM-generated conclusion.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    observation_id: str = Field(
        min_length=1,
        description="Stable identifier for this observation.",
    )

    type: str = Field(
        min_length=1,
        description="Observation category.",
    )

    variables: list[str] = Field(
        default_factory=list,
        description="Dataset variables involved in the observation.",
    )

    statistic: dict[str, Any] = Field(
        default_factory=dict,
        description="Numerical or structured evidence.",
    )

    sample_size: int | None = Field(
        default=None,
        ge=0,
        description="Number of observations used in the calculation.",
    )

    method: str | list[str] | None = Field(
        default=None,
        description="Analytical method used to produce the observation.",
    )

    source: str = Field(
        min_length=1,
        description="Engine that produced the observation.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional deterministic metadata.",
    )

    @field_validator("observation_id")
    @classmethod
    def normalize_observation_id(
        cls,
        value: str,
    ) -> str:
        return value.strip()

    @field_validator("type")
    @classmethod
    def normalize_type(
        cls,
        value: str,
    ) -> str:
        return value.strip().lower()

    @field_validator("source")
    @classmethod
    def normalize_source(
        cls,
        value: str,
    ) -> str:
        return value.strip().lower()

    @field_validator("variables")
    @classmethod
    def normalize_variables(
        cls,
        value: list[str],
    ) -> list[str]:
        return [
            variable.strip()
            for variable in value
            if variable.strip()
        ]

    def to_dict(self) -> dict[str, Any]:
        """
        Return the observation as a standard Python dictionary.
        """

        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """
        Return deterministic JSON representation.
        """

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )


class ObservationStore:
    """
    In-memory observation store for the NEXUS-6 analysis pipeline.

    The store is intentionally simple in Phase 1. It provides a clean
    interface that can later be backed by a persistent database without
    changing the analytical engines.
    """

    def __init__(self) -> None:
        self._observations: dict[
            str,
            Observation,
        ] = {}

    def add(
        self,
        observation: Observation,
    ) -> Observation:
        """
        Add an observation to the store.

        Adding the same observation ID again with identical content is
        treated as idempotent.

        Reusing an observation ID for different content raises an error.
        """

        existing = self._observations.get(
            observation.observation_id
        )

        if existing is None:
            self._observations[
                observation.observation_id
            ] = observation

            return observation

        if existing.to_json() == observation.to_json():
            return existing

        raise ValueError(
            "Observation ID already exists with different content: "
            f"{observation.observation_id}"
        )

    def add_many(
        self,
        observations: list[Observation],
    ) -> list[Observation]:
        """
        Add multiple observations in order.
        """

        stored: list[Observation] = []

        for observation in observations:
            stored.append(
                self.add(observation)
            )

        return stored

    def get(
        self,
        observation_id: str,
    ) -> Observation | None:
        """
        Retrieve one observation by ID.
        """

        return self._observations.get(
            observation_id
        )

    def require(
        self,
        observation_id: str,
    ) -> Observation:
        """
        Retrieve one observation or raise a clear error.
        """

        observation = self.get(
            observation_id
        )

        if observation is None:
            raise KeyError(
                "Observation not found: "
                f"{observation_id}"
            )

        return observation

    def all(self) -> list[Observation]:
        """
        Return all observations in insertion order.
        """

        return list(
            self._observations.values()
        )

    def filter(
        self,
        *,
        observation_type: str | None = None,
        source: str | None = None,
    ) -> list[Observation]:
        """
        Filter observations by type and/or source.
        """

        normalized_type = (
            observation_type.strip().lower()
            if observation_type is not None
            else None
        )

        normalized_source = (
            source.strip().lower()
            if source is not None
            else None
        )

        results: list[Observation] = []

        for observation in self._observations.values():
            if (
                normalized_type is not None
                and observation.type != normalized_type
            ):
                continue

            if (
                normalized_source is not None
                and observation.source != normalized_source
            ):
                continue

            results.append(observation)

        return results

    def remove(
        self,
        observation_id: str,
    ) -> bool:
        """
        Remove an observation.

        Returns True when an observation existed and was removed.
        """

        if observation_id not in self._observations:
            return False

        del self._observations[
            observation_id
        ]

        return True

    def clear(self) -> None:
        """
        Remove all observations.
        """

        self._observations.clear()

    def count(self) -> int:
        """
        Return the number of stored observations.
        """

        return len(
            self._observations
        )

    def to_dict(self) -> list[dict[str, Any]]:
        """
        Return all observations as dictionaries.
        """

        return [
            observation.to_dict()
            for observation in self._observations.values()
        ]

    def to_json(self) -> str:
        """
        Return the complete store as deterministic JSON.
        """

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )


def create_observation_id(
    *,
    observation_type: str,
    variables: list[str],
    source: str,
    method: str | list[str] | None = None,
) -> str:
    """
    Create a deterministic observation ID from observation identity.

    The generated ID remains stable when the same analytical identity
    is encountered again.
    """

    identity = {
        "type": observation_type.strip().lower(),
        "variables": variables,
        "source": source.strip().lower(),
        "method": method,
    }

    serialized = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()[:16]

    return f"obs_{digest}"