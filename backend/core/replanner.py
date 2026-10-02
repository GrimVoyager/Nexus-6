from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from backend.core.critic import (
    CriticDecision,
    CriticStatus,
)
from backend.core.planner import (
    AnalysisTier,
    ExecutionPlan,
    TierStatus,
)


class ReplanState(str, Enum):
    INITIAL = "INITIAL"
    REPLANNING = "REPLANNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class ReplanDecision(BaseModel):
    """
    Structured decision produced by the NEXUS-6 replanner.

    The replanner converts critic findings into a revised
    execution plan without modifying the source dataset.
    """

    state: ReplanState
    replan_required: bool
    replan_count: int
    max_replans: int
    reason: str
    requested_tiers: list[int] = Field(
        default_factory=list
    )
    excluded_tiers: list[int] = Field(
        default_factory=list
    )
    changes: list[str] = Field(
        default_factory=list
    )


class Replanner:
    """
    Deterministic replanning layer for NEXUS-6.

    Responsibilities:
        - Read the critic decision.
        - Determine whether another execution cycle is required.
        - Identify the tiers that should be revisited.
        - Produce a revised ExecutionPlan.
        - Enforce a maximum replanning limit.

    The replanner does not:
        - modify the source dataset
        - interpret raw numerical results
        - execute analysis
        - use an LLM
    """

    def __init__(
        self,
        max_replans: int = 3,
    ) -> None:
        if max_replans < 0:
            raise ValueError(
                "max_replans cannot be negative."
            )

        self.max_replans = max_replans

    def replan(
        self,
        *,
        current_plan: ExecutionPlan,
        critic_decision: CriticDecision,
        replan_count: int,
    ) -> tuple[
        ReplanDecision,
        ExecutionPlan | None,
    ]:
        if replan_count < 0:
            raise ValueError(
                "replan_count cannot be negative."
            )

        if replan_count > self.max_replans:
            raise ValueError(
                "replan_count cannot exceed max_replans."
            )

        if critic_decision.status == CriticStatus.SUFFICIENT:
            decision = ReplanDecision(
                state=ReplanState.COMPLETE,
                replan_required=False,
                replan_count=replan_count,
                max_replans=self.max_replans,
                reason=(
                    "Critic determined that the available "
                    "evidence is sufficient."
                ),
                requested_tiers=[],
                excluded_tiers=[],
                changes=[],
            )

            return decision, None

        if not critic_decision.replan_required:
            decision = ReplanDecision(
                state=ReplanState.COMPLETE,
                replan_required=False,
                replan_count=replan_count,
                max_replans=self.max_replans,
                reason=(
                    "Critic did not request another execution cycle."
                ),
                requested_tiers=[],
                excluded_tiers=[],
                changes=[],
            )

            return decision, None

        if replan_count >= self.max_replans:
            decision = ReplanDecision(
                state=ReplanState.FAILED,
                replan_required=False,
                replan_count=replan_count,
                max_replans=self.max_replans,
                reason=(
                    "Maximum replanning limit has been reached."
                ),
                requested_tiers=[],
                excluded_tiers=[],
                changes=[],
            )

            return decision, None

        requested_tiers = self._determine_requested_tiers(
            critic_decision
        )

        revised_plan = self._build_revised_plan(
            current_plan=current_plan,
            critic_decision=critic_decision,
            requested_tiers=requested_tiers,
        )

        changes = self._describe_changes(
            current_plan=current_plan,
            revised_plan=revised_plan,
        )

        decision = ReplanDecision(
            state=ReplanState.REPLANNING,
            replan_required=True,
            replan_count=replan_count + 1,
            max_replans=self.max_replans,
            reason=self._build_reason(
                critic_decision
            ),
            requested_tiers=requested_tiers,
            excluded_tiers=[],
            changes=changes,
        )

        return decision, revised_plan

    @staticmethod
    def _determine_requested_tiers(
        critic_decision: CriticDecision,
    ) -> list[int]:
        requested: list[int] = []

        for requirement in (
            critic_decision.missing_requirements
        ):
            if requirement == "query_execution":
                requested.append(
                    int(AnalysisTier.QUERY)
                )

            elif requirement == "repair_analysis":
                requested.append(
                    int(AnalysisTier.REPAIR)
                )

            elif requirement == "intelligence_analysis":
                requested.append(
                    int(AnalysisTier.INTELLIGENCE)
                )

            elif requirement == "engineering_analysis":
                requested.append(
                    int(AnalysisTier.ENGINEERING)
                )

            elif requirement == "at_least_one_observation":
                requested.extend(
                    [
                        int(AnalysisTier.OVERVIEW),
                        int(AnalysisTier.AUDIT),
                    ]
                )

        if (
            critic_decision.status
            == CriticStatus.CONTRADICTORY
        ):
            requested.append(
                int(AnalysisTier.INTELLIGENCE)
            )

        return sorted(
            set(requested)
        )

    def _build_revised_plan(
        self,
        *,
        current_plan: ExecutionPlan,
        critic_decision: CriticDecision,
        requested_tiers: list[int],
    ) -> ExecutionPlan:
        decisions = []

        for decision in current_plan.tier_decisions:
            tier_number = int(
                decision.tier
            )

            if tier_number in requested_tiers:
                decisions.append(
                    decision.model_copy(
                        update={
                            "status": TierStatus.EXECUTE.value,
                            "reason": (
                                "Replanned after critic feedback: "
                                + self._build_reason(
                                    critic_decision
                                )
                            ),
                        }
                    )
                )
            else:
                decisions.append(
                    decision
                )

        return current_plan.model_copy(
            update={
                "tier_decisions": decisions,
                "reasoning": (
                    current_plan.reasoning
                    + " Revised after critic feedback: "
                    + self._build_reason(
                        critic_decision
                    )
                ),
            }
        )

    @staticmethod
    def _describe_changes(
        *,
        current_plan: ExecutionPlan,
        revised_plan: ExecutionPlan,
    ) -> list[str]:
        changes: list[str] = []

        current_by_tier = {
            int(decision.tier): decision
            for decision in current_plan.tier_decisions
        }

        revised_by_tier = {
            int(decision.tier): decision
            for decision in revised_plan.tier_decisions
        }

        for tier in sorted(
            revised_by_tier
        ):
            current = current_by_tier[tier]
            revised = revised_by_tier[tier]

            if (
                current.status
                != revised.status
            ):
                changes.append(
                    f"Tier {tier}: "
                    f"{current.status} -> "
                    f"{revised.status}"
                )

        if not changes:
            changes.append(
                "No tier status changed."
            )

        return changes

    @staticmethod
    def _build_reason(
        critic_decision: CriticDecision,
    ) -> str:
        if critic_decision.reasons:
            return " ".join(
                critic_decision.reasons
            )

        return (
            "Critic requested another execution cycle."
        )