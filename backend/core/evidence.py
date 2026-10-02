from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.core.intelligence import DatasetIntelligence
from backend.core.observation import Observation
from backend.core.observation_collector import ObservationCollector


class EvidenceSummary(BaseModel):
    """
    Deterministic summary of the evidence package.

    This summary describes what evidence exists. It does not interpret
    the business meaning of that evidence.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    observation_count: int = Field(
        default=0,
        ge=0,
    )

    observation_types: list[str] = Field(
        default_factory=list,
    )

    observation_sources: list[str] = Field(
        default_factory=list,
    )

    source_counts: dict[str, int] = Field(
        default_factory=dict,
    )

    type_counts: dict[str, int] = Field(
        default_factory=dict,
    )

    @field_validator("observation_types")
    @classmethod
    def normalize_observation_types(
        cls,
        value: list[str],
    ) -> list[str]:
        return sorted(
            {
                item.strip().lower()
                for item in value
                if item.strip()
            }
        )

    @field_validator("observation_sources")
    @classmethod
    def normalize_observation_sources(
        cls,
        value: list[str],
    ) -> list[str]:
        return sorted(
            {
                item.strip().lower()
                for item in value
                if item.strip()
            }
        )


class DatasetEvidencePackage(BaseModel):
    """
    Unified evidence contract for the NEXUS-6 analytical pipeline.

    The package combines:
        - Dataset intelligence
        - Deterministic observations
        - Evidence summary
        - Dataset metadata

    The package contains evidence and facts only. Interpretation,
    recommendations, and business conclusions belong to later layers.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    package_version: str = Field(
        default="1.0",
        min_length=1,
    )

    table_name: str = Field(
        min_length=1,
    )

    row_count: int = Field(
        ge=0,
    )

    column_count: int = Field(
        ge=0,
    )

    intelligence: dict[str, Any] = Field(
        default_factory=dict,
    )

    observations: list[Observation] = Field(
        default_factory=list,
    )

    summary: EvidenceSummary = Field(
        default_factory=EvidenceSummary,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("table_name")
    @classmethod
    def normalize_table_name(
        cls,
        value: str,
    ) -> str:
        return value.strip()

    def to_dict(self) -> dict[str, Any]:
        """
        Return the complete evidence package as a Python dictionary.
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

    def get_observations_by_type(
        self,
        observation_type: str,
    ) -> list[Observation]:
        """
        Return observations matching one observation type.
        """

        normalized_type = (
            observation_type.strip().lower()
        )

        return [
            observation
            for observation in self.observations
            if observation.type == normalized_type
        ]

    def get_observations_by_source(
        self,
        source: str,
    ) -> list[Observation]:
        """
        Return observations produced by one analytical engine.
        """

        normalized_source = (
            source.strip().lower()
        )

        return [
            observation
            for observation in self.observations
            if observation.source == normalized_source
        ]


class DatasetEvidenceBuilder:
    """
    Builds one DatasetEvidencePackage from the existing NEXUS-6
    intelligence and observation layers.
    """

    def __init__(
        self,
        intelligence: DatasetIntelligence,
        observation_collector: ObservationCollector,
    ) -> None:
        self.intelligence = intelligence
        self.observation_collector = (
            observation_collector
        )

    def build(self) -> DatasetEvidencePackage:
        """
        Build the complete evidence package.
        """

        intelligence_profile = (
            self.intelligence.build_profile()
        )

        observation_store = (
            self.observation_collector.collect()
        )

        observations = observation_store.all()

        summary = self._build_summary(
            observations
        )

        return DatasetEvidencePackage(
            package_version="1.0",
            table_name=intelligence_profile[
                "table_name"
            ],
            row_count=intelligence_profile[
                "row_count"
            ],
            column_count=intelligence_profile[
                "column_count"
            ],
            intelligence=intelligence_profile,
            observations=observations,
            summary=summary,
            metadata={
                "observation_store_count": (
                    observation_store.count()
                ),
                "evidence_sources": (
                    summary.observation_sources
                ),
            },
        )

    @staticmethod
    def _build_summary(
        observations: list[Observation],
    ) -> EvidenceSummary:
        source_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}

        for observation in observations:
            source_counts[observation.source] = (
                source_counts.get(
                    observation.source,
                    0,
                )
                + 1
            )

            type_counts[observation.type] = (
                type_counts.get(
                    observation.type,
                    0,
                )
                + 1
            )

        return EvidenceSummary(
            observation_count=len(observations),
            observation_types=list(
                type_counts.keys()
            ),
            observation_sources=list(
                source_counts.keys()
            ),
            source_counts=source_counts,
            type_counts=type_counts,
        )