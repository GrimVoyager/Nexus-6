from __future__ import annotations

import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


DomainType = Literal[
    "hr_workforce",
    "bfsi_finance",
    "healthcare",
    "retail_ecom",
    "general",
]


class TierStatus(str, Enum):
    """
    Execution status for one NEXUS-6 analytical tier.
    """

    EXECUTE = "EXECUTE"
    PARTIAL = "PARTIAL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class AnalysisTier(int, Enum):
    """
    Universal NEXUS-6 analysis tiers.
    """

    QUERY = 1
    OVERVIEW = 2
    AUDIT = 3
    REPAIR = 4
    INTELLIGENCE = 5
    ENGINEERING = 6


class TierDecision(BaseModel):
    """
    Decision describing how one universal NEXUS-6 tier should be handled.

    The decision is descriptive. It does not execute the tier.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
    )

    tier: AnalysisTier = Field(
        description="Universal NEXUS-6 tier number.",
    )

    name: str = Field(
        min_length=1,
        description="Human-readable tier name.",
    )

    status: TierStatus = Field(
        description="Execution status for the tier.",
    )

    reason: str = Field(
        min_length=1,
        description="Deterministic reason for the selected status.",
    )

    @field_validator("name")
    @classmethod
    def normalize_name(
        cls,
        value: str,
    ) -> str:
        return value.strip()

    @field_validator("reason")
    @classmethod
    def normalize_reason(
        cls,
        value: str,
    ) -> str:
        return value.strip()


class ExecutionPlan(BaseModel):
    """
    Typed execution contract produced by the NEXUS-6 orchestrator.

    The planner describes what should happen. It does not perform
    analytical calculations.

    All six universal tiers are represented explicitly. Each tier may
    be executed fully, partially, or marked not applicable.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
    )

    selected_tiers: list[int] = Field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6],
        description=(
            "Universal NEXUS-6 tiers evaluated by the orchestrator."
        ),
    )

    tier_decisions: list[TierDecision] = Field(
        default_factory=list,
        description=(
            "Explicit execution decision for every universal tier."
        ),
    )

    detected_domain: DomainType = Field(
        default="general",
        description="Detected business or analytical domain.",
    )

    overlay: bool = Field(
        default=False,
        description=(
            "Whether a domain-specific analytical overlay should "
            "be applied."
        ),
    )

    primary_sql: str | None = Field(
        default=None,
        description=(
            "Optional validated SQL statement representing the primary "
            "analytical request."
        ),
    )

    target: str | None = Field(
        default=None,
        description=(
            "Optional target variable identified for predictive or "
            "supervised analysis."
        ),
    )

    reasoning: str = Field(
        default="",
        description=(
            "Concise explanation of why the selected domain and plan "
            "were chosen."
        ),
    )

    @field_validator("selected_tiers")
    @classmethod
    def validate_selected_tiers(
        cls,
        value: list[int],
    ) -> list[int]:
        if value != [1, 2, 3, 4, 5, 6]:
            raise ValueError(
                "selected_tiers must contain exactly "
                "[1, 2, 3, 4, 5, 6]"
            )

        return value

    @field_validator("tier_decisions")
    @classmethod
    def validate_tier_decisions(
        cls,
        value: list[TierDecision],
    ) -> list[TierDecision]:
        if not value:
            return value

        tier_numbers = [
            int(decision.tier)
            for decision in value
        ]

        if len(tier_numbers) != 6:
            raise ValueError(
                "tier_decisions must contain exactly "
                "six tier decisions"
            )

        if sorted(tier_numbers) != [
            1,
            2,
            3,
            4,
            5,
            6,
        ]:
            raise ValueError(
                "tier_decisions must contain exactly "
                "tiers 1 through 6"
            )

        if len(set(tier_numbers)) != 6:
            raise ValueError(
                "tier_decisions cannot contain duplicate tiers"
            )

        return value

    @field_validator("primary_sql")
    @classmethod
    def validate_primary_sql(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = value.strip()

        if not normalized:
            return None

        if not normalized.lower().startswith("select"):
            raise ValueError(
                "primary_sql must currently be a SELECT statement"
            )

        return normalized

    @field_validator("reasoning")
    @classmethod
    def normalize_reasoning(
        cls,
        value: str,
    ) -> str:
        return value.strip()

    def get_tier_decision(
        self,
        tier: int | AnalysisTier,
    ) -> TierDecision | None:
        """
        Retrieve the decision for one tier.
        """

        tier_number = int(tier)

        for decision in self.tier_decisions:
            if int(decision.tier) == tier_number:
                return decision

        return None

    def to_dict(self) -> dict:
        """
        Return the execution plan as a JSON-compatible dictionary.
        """

        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """
        Return a deterministic JSON representation.
        """

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )


def create_default_tier_decisions() -> list[TierDecision]:
    """
    Create a neutral planning template.

    No analytical assumption is made here. Every tier starts as
    NOT_APPLICABLE until the orchestrator has enough evidence to
    determine otherwise.
    """

    return [
        TierDecision(
            tier=AnalysisTier.QUERY,
            name="Query",
            status=TierStatus.NOT_APPLICABLE,
            reason=(
                "No query-specific objective has been provided."
            ),
        ),
        TierDecision(
            tier=AnalysisTier.OVERVIEW,
            name="Overview",
            status=TierStatus.NOT_APPLICABLE,
            reason=(
                "No overview decision has been made yet."
            ),
        ),
        TierDecision(
            tier=AnalysisTier.AUDIT,
            name="Audit",
            status=TierStatus.NOT_APPLICABLE,
            reason=(
                "No audit decision has been made yet."
            ),
        ),
        TierDecision(
            tier=AnalysisTier.REPAIR,
            name="Repair",
            status=TierStatus.NOT_APPLICABLE,
            reason=(
                "No repair requirement has been established."
            ),
        ),
        TierDecision(
            tier=AnalysisTier.INTELLIGENCE,
            name="Intelligence",
            status=TierStatus.NOT_APPLICABLE,
            reason=(
                "No intelligence decision has been made yet."
            ),
        ),
        TierDecision(
            tier=AnalysisTier.ENGINEERING,
            name="Engineering",
            status=TierStatus.NOT_APPLICABLE,
            reason=(
                "No engineering or migration requirement "
                "has been established."
            ),
        ),
    ]


def create_execution_plan(
    *,
    detected_domain: DomainType = "general",
    overlay: bool = False,
    primary_sql: str | None = None,
    target: str | None = None,
    reasoning: str = "",
    tier_decisions: list[TierDecision] | None = None,
) -> ExecutionPlan:
    """
    Create a validated NEXUS-6 execution plan.
    """

    decisions = (
        create_default_tier_decisions()
        if tier_decisions is None
        else tier_decisions
    )

    return ExecutionPlan(
        selected_tiers=[
            1,
            2,
            3,
            4,
            5,
            6,
        ],
        tier_decisions=decisions,
        detected_domain=detected_domain,
        overlay=overlay,
        primary_sql=primary_sql,
        target=target,
        reasoning=reasoning,
    )