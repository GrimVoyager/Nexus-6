from __future__ import annotations

from typing import Any

from backend.core.evidence import (
    DatasetEvidencePackage,
    EvidenceSummary,
)
from backend.core.executor import TierExecutionResult
from backend.core.observation import Observation


class EvidenceMerger:
    """
    Merges execution-produced evidence into an existing
    DatasetEvidencePackage.

    The merger:
        - Preserves the original evidence package.
        - Adds new observations produced by execution.
        - Prevents conflicting observation IDs.
        - Rebuilds evidence summaries.
        - Records execution provenance.

    The merger does not interpret observations and does not
    modify the source dataset.
    """

    def merge(
        self,
        *,
        original: DatasetEvidencePackage,
        execution_result: TierExecutionResult,
    ) -> DatasetEvidencePackage:
        original_observations = list(
            original.observations
        )

        merged_observations = list(
            original_observations
        )

        existing_by_id = {
            observation.observation_id: observation
            for observation in original_observations
        }

        added_observation_count = 0

        for observation in execution_result.observations.all():
            existing = existing_by_id.get(
                observation.observation_id
            )

            if existing is None:
                merged_observations.append(
                    observation
                )

                existing_by_id[
                    observation.observation_id
                ] = observation

                added_observation_count += 1

                continue

            if existing.to_json() != observation.to_json():
                raise ValueError(
                    "Execution produced an observation with an "
                    "existing ID but different content: "
                    f"{observation.observation_id}"
                )

        summary = self._build_summary(
            merged_observations
        )

        metadata = dict(
            original.metadata
        )

        existing_merge_metadata = metadata.get(
            "evidence_merge"
        )

        if (
            existing_merge_metadata is not None
            and added_observation_count == 0
        ):
            metadata["evidence_merge"] = (
                existing_merge_metadata
            )
        else:
            metadata["evidence_merge"] = {
                "performed": True,
                "original_observation_count": len(
                    original_observations
                ),
                "execution_observation_count": (
                    execution_result.observation_count
                ),
                "added_observation_count": (
                    added_observation_count
                ),
                "merged_observation_count": len(
                    merged_observations
                ),
                "executed_tiers": list(
                    execution_result.executed_tiers
                ),
                "partial_tiers": list(
                    execution_result.partial_tiers
                ),
                "skipped_tiers": list(
                    execution_result.skipped_tiers
                ),
            }

        return DatasetEvidencePackage(
            package_version=original.package_version,
            table_name=original.table_name,
            row_count=original.row_count,
            column_count=original.column_count,
            intelligence=dict(
                original.intelligence
            ),
            observations=merged_observations,
            summary=summary,
            metadata=metadata,
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