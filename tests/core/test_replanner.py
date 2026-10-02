from __future__ import annotations

from backend.core.critic import (
    CriticDecision,
    CriticStatus,
)
from backend.core.planner import (
    ExecutionPlan,
    create_execution_plan,
)
from backend.core.replanner import (
    ReplanState,
    Replanner,
)


def create_critic_decision(
    *,
    status: CriticStatus,
    replan_required: bool,
    missing_requirements: list[str] | None = None,
    reasons: list[str] | None = None,
) -> CriticDecision:
    return CriticDecision(
        status=status,
        replan_required=replan_required,
        reasons=reasons or [],
        evidence_count=1,
        executed_tiers=[2, 3],
        partial_tiers=[],
        skipped_tiers=[1, 4, 5, 6],
        missing_requirements=(
            missing_requirements or []
        ),
        contradiction_count=0,
    )


def test_sufficient_does_not_replan() -> None:
    plan = create_execution_plan(
        detected_domain="general",
        overlay=False,
        primary_sql=None,
        target=None,
        reasoning="Initial test execution plan.",
    )

    critic = create_critic_decision(
        status=CriticStatus.SUFFICIENT,
        replan_required=False,
    )

    decision, revised_plan = Replanner().replan(
        current_plan=plan,
        critic_decision=critic,
        replan_count=0,
    )

    assert (
        decision.state
        == ReplanState.COMPLETE
    )

    assert decision.replan_required is False
    assert revised_plan is None


def test_query_requirement_triggers_replan() -> None:
    plan = create_execution_plan(
        detected_domain="general",
        overlay=False,
        primary_sql=None,
        target=None,
        reasoning="Initial test execution plan.",
    )

    critic = create_critic_decision(
        status=CriticStatus.INSUFFICIENT,
        replan_required=True,
        missing_requirements=[
            "query_execution"
        ],
        reasons=[
            "Query execution is missing."
        ],
    )

    decision, revised_plan = Replanner().replan(
        current_plan=plan,
        critic_decision=critic,
        replan_count=0,
    )

    assert (
        decision.state
        == ReplanState.REPLANNING
    )

    assert decision.replan_required is True
    assert decision.replan_count == 1

    assert (
        1
        in decision.requested_tiers
    )

    assert revised_plan is not None


def test_repair_requirement_triggers_replan() -> None:
    plan = create_execution_plan(
        detected_domain="general",
        overlay=False,
        primary_sql=None,
        target=None,
        reasoning="Initial test execution plan.",
    )

    critic = create_critic_decision(
        status=CriticStatus.INSUFFICIENT,
        replan_required=True,
        missing_requirements=[
            "repair_analysis"
        ],
    )

    decision, revised_plan = Replanner().replan(
        current_plan=plan,
        critic_decision=critic,
        replan_count=0,
    )

    assert (
        4
        in decision.requested_tiers
    )

    assert revised_plan is not None


def test_intelligence_requirement_triggers_replan() -> None:
    plan = create_execution_plan(
        detected_domain="general",
        overlay=False,
        primary_sql=None,
        target=None,
        reasoning="Initial test execution plan.",
    )

    critic = create_critic_decision(
        status=CriticStatus.INSUFFICIENT,
        replan_required=True,
        missing_requirements=[
            "intelligence_analysis"
        ],
    )

    decision, revised_plan = Replanner().replan(
        current_plan=plan,
        critic_decision=critic,
        replan_count=0,
    )

    assert (
        5
        in decision.requested_tiers
    )

    assert revised_plan is not None


def test_engineering_requirement_triggers_replan() -> None:
    plan = create_execution_plan(
        detected_domain="general",
        overlay=False,
        primary_sql=None,
        target=None,
        reasoning="Initial test execution plan.",
    )

    critic = create_critic_decision(
        status=CriticStatus.INSUFFICIENT,
        replan_required=True,
        missing_requirements=[
            "engineering_analysis"
        ],
    )

    decision, revised_plan = Replanner().replan(
        current_plan=plan,
        critic_decision=critic,
        replan_count=0,
    )

    assert (
        6
        in decision.requested_tiers
    )

    assert revised_plan is not None


def test_empty_evidence_triggers_overview_and_audit() -> None:
    plan = create_execution_plan(
        detected_domain="general",
        overlay=False,
        primary_sql=None,
        target=None,
        reasoning="Initial test execution plan.",
    )

    critic = create_critic_decision(
        status=CriticStatus.INSUFFICIENT,
        replan_required=True,
        missing_requirements=[
            "at_least_one_observation"
        ],
    )

    decision, revised_plan = Replanner().replan(
        current_plan=plan,
        critic_decision=critic,
        replan_count=0,
    )

    assert (
        2
        in decision.requested_tiers
    )

    assert (
        3
        in decision.requested_tiers
    )

    assert revised_plan is not None


def test_contradiction_triggers_intelligence_recheck() -> None:
    plan = create_execution_plan(
        detected_domain="general",
        overlay=False,
        primary_sql=None,
        target=None,
        reasoning="Initial test execution plan.",
    )

    critic = create_critic_decision(
        status=CriticStatus.CONTRADICTORY,
        replan_required=True,
        reasons=[
            "Conflicting observations detected."
        ],
    )

    decision, revised_plan = Replanner().replan(
        current_plan=plan,
        critic_decision=critic,
        replan_count=0,
    )

    assert (
        5
        in decision.requested_tiers
    )

    assert revised_plan is not None


def test_max_replans_protects_against_infinite_loop() -> None:
    plan = create_execution_plan(
        detected_domain="general",
        overlay=False,
        primary_sql=None,
        target=None,
        reasoning="Initial test execution plan.",
    )

    critic = create_critic_decision(
        status=CriticStatus.INSUFFICIENT,
        replan_required=True,
        missing_requirements=[
            "intelligence_analysis"
        ],
    )

    replanner = Replanner(
        max_replans=3
    )

    decision, revised_plan = replanner.replan(
        current_plan=plan,
        critic_decision=critic,
        replan_count=3,
    )

    assert (
        decision.state
        == ReplanState.FAILED
    )

    assert decision.replan_required is False
    assert revised_plan is None


def test_replan_count_increments() -> None:
    plan = create_execution_plan(
        detected_domain="general",
        overlay=False,
        primary_sql=None,
        target=None,
        reasoning="Initial test execution plan.",
    )

    critic = create_critic_decision(
        status=CriticStatus.INSUFFICIENT,
        replan_required=True,
        missing_requirements=[
            "repair_analysis"
        ],
    )

    replanner = Replanner(
        max_replans=3
    )

    decision, _ = replanner.replan(
        current_plan=plan,
        critic_decision=critic,
        replan_count=1,
    )

    assert decision.replan_count == 2


def test_revised_plan_preserves_six_tier_structure() -> None:
    plan = create_execution_plan(
        detected_domain="general",
        overlay=False,
        primary_sql=None,
        target=None,
        reasoning="Initial test execution plan.",
    )

    critic = create_critic_decision(
        status=CriticStatus.INSUFFICIENT,
        replan_required=True,
        missing_requirements=[
            "intelligence_analysis"
        ],
    )

    decision, revised_plan = Replanner().replan(
        current_plan=plan,
        critic_decision=critic,
        replan_count=0,
    )

    assert decision.replan_required is True
    assert revised_plan is not None

    assert (
        len(revised_plan.tier_decisions)
        == 6
    )

    tiers = [
        int(item.tier)
        for item in revised_plan.tier_decisions
    ]

    assert tiers == [1, 2, 3, 4, 5, 6]


def test_negative_replan_count_is_rejected() -> None:
    plan = create_execution_plan(
        detected_domain="general",
        overlay=False,
        primary_sql=None,
        target=None,
        reasoning="Initial test execution plan.",
    )

    critic = create_critic_decision(
        status=CriticStatus.SUFFICIENT,
        replan_required=False,
    )

    try:
        Replanner().replan(
            current_plan=plan,
            critic_decision=critic,
            replan_count=-1,
        )
    except ValueError as error:
        assert (
            "cannot be negative"
            in str(error)
        )
    else:
        raise AssertionError(
            "Negative replan count was accepted."
        )


def main() -> None:
    print("=" * 60)
    print("NEXUS-6 REPLANNER TEST")
    print("=" * 60)

    test_sufficient_does_not_replan()
    print("PASS: Sufficient evidence stops replanning")

    test_query_requirement_triggers_replan()
    print("PASS: Query requirement replanning")

    test_repair_requirement_triggers_replan()
    print("PASS: Repair requirement replanning")

    test_intelligence_requirement_triggers_replan()
    print("PASS: Intelligence requirement replanning")

    test_engineering_requirement_triggers_replan()
    print("PASS: Engineering requirement replanning")

    test_empty_evidence_triggers_overview_and_audit()
    print("PASS: Empty evidence recovery")

    test_contradiction_triggers_intelligence_recheck()
    print("PASS: Contradiction recovery")

    test_max_replans_protects_against_infinite_loop()
    print("PASS: Maximum replan protection")

    test_replan_count_increments()
    print("PASS: Replan count tracking")

    test_revised_plan_preserves_six_tier_structure()
    print("PASS: Six-tier plan preservation")

    test_negative_replan_count_is_rejected()
    print("PASS: Invalid replan count protection")

    print()
    print("=" * 60)
    print("ALL NEXUS-6 REPLANNER TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()