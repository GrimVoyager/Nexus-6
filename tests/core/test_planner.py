from __future__ import annotations

import json

from backend.core.planner import (
    AnalysisTier,
    ExecutionPlan,
    TierDecision,
    TierStatus,
    create_default_tier_decisions,
    create_execution_plan,
)


def test_default_tier_decisions() -> None:
    decisions = create_default_tier_decisions()

    assert len(decisions) == 6

    tiers = [
        int(decision.tier)
        for decision in decisions
    ]

    assert tiers == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]

    for decision in decisions:
        assert (
            decision.status
            == TierStatus.NOT_APPLICABLE.value
        )

        assert decision.reason


def test_tier_decision_model() -> None:
    decision = TierDecision(
        tier=AnalysisTier.OVERVIEW,
        name="Overview",
        status=TierStatus.EXECUTE,
        reason=(
            "The dataset contains sufficient rows and "
            "columns for overview analysis."
        ),
    )

    assert int(decision.tier) == 2
    assert decision.name == "Overview"
    assert decision.status == "EXECUTE"
    assert decision.reason


def test_execution_plan_contains_all_six_tiers() -> None:
    decisions = [
        TierDecision(
            tier=AnalysisTier.QUERY,
            name="Query",
            status=TierStatus.EXECUTE,
            reason="The user supplied a query objective.",
        ),
        TierDecision(
            tier=AnalysisTier.OVERVIEW,
            name="Overview",
            status=TierStatus.EXECUTE,
            reason="Dataset overview is applicable.",
        ),
        TierDecision(
            tier=AnalysisTier.AUDIT,
            name="Audit",
            status=TierStatus.EXECUTE,
            reason="Quality evidence is available.",
        ),
        TierDecision(
            tier=AnalysisTier.REPAIR,
            name="Repair",
            status=TierStatus.PARTIAL,
            reason="Repair analysis is possible but mutation is disabled.",
        ),
        TierDecision(
            tier=AnalysisTier.INTELLIGENCE,
            name="Intelligence",
            status=TierStatus.EXECUTE,
            reason="Analytical intelligence is applicable.",
        ),
        TierDecision(
            tier=AnalysisTier.ENGINEERING,
            name="Engineering",
            status=TierStatus.NOT_APPLICABLE,
            reason="No migration or engineering objective exists.",
        ),
    ]

    plan = create_execution_plan(
        detected_domain="hr_workforce",
        overlay=True,
        target="Performance",
        reasoning=(
            "The dataset contains workforce information "
            "and supports analytical intelligence."
        ),
        tier_decisions=decisions,
    )

    assert plan.selected_tiers == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]

    assert len(plan.tier_decisions) == 6

    assert (
        plan.detected_domain
        == "hr_workforce"
    )

    assert plan.overlay is True
    assert plan.target == "Performance"

    assert (
        plan.get_tier_decision(1).status
        == "EXECUTE"
    )

    assert (
        plan.get_tier_decision(4).status
        == "PARTIAL"
    )

    assert (
        plan.get_tier_decision(6).status
        == "NOT_APPLICABLE"
    )


def test_execution_plan_json_serialization() -> None:
    plan = create_execution_plan(
        detected_domain="general",
        reasoning="Test execution plan.",
    )

    serialized = plan.to_json()

    parsed = json.loads(
        serialized
    )

    assert isinstance(parsed, dict)

    assert (
        parsed["selected_tiers"]
        == [1, 2, 3, 4, 5, 6]
    )

    assert len(
        parsed["tier_decisions"]
    ) == 6

    assert (
        parsed["tier_decisions"][0]["status"]
        == "NOT_APPLICABLE"
    )


def test_execution_plan_round_trip() -> None:
    plan = create_execution_plan(
        detected_domain="bfsi_finance",
        overlay=True,
        primary_sql=(
            "SELECT * FROM dataset"
        ),
        target="Loan_Status",
        reasoning=(
            "Finance dataset with a supervised "
            "learning target."
        ),
    )

    serialized = plan.to_json()

    restored = (
        ExecutionPlan.model_validate_json(
            serialized
        )
    )

    assert (
        restored.detected_domain
        == plan.detected_domain
    )

    assert (
        restored.overlay
        == plan.overlay
    )

    assert (
        restored.primary_sql
        == plan.primary_sql
    )

    assert (
        restored.target
        == plan.target
    )

    assert len(
        restored.tier_decisions
    ) == 6


def test_invalid_tier_decisions_are_rejected() -> None:
    invalid_decisions = [
        TierDecision(
            tier=AnalysisTier.QUERY,
            name="Query",
            status=TierStatus.EXECUTE,
            reason="Valid first tier.",
        )
    ]

    try:
        ExecutionPlan(
            selected_tiers=[
                1,
                2,
                3,
                4,
                5,
                6,
            ],
            tier_decisions=invalid_decisions,
            detected_domain="general",
        )
    except ValueError as error:
        assert "six tier decisions" in str(error)
    else:
        raise AssertionError(
            "Expected invalid tier decision list to be rejected."
        )


def test_invalid_tier_number_is_rejected() -> None:
    invalid_decisions = [
        TierDecision(
            tier=AnalysisTier.QUERY,
            name="Query",
            status=TierStatus.EXECUTE,
            reason="Valid.",
        ),
        TierDecision(
            tier=AnalysisTier.OVERVIEW,
            name="Overview",
            status=TierStatus.EXECUTE,
            reason="Valid.",
        ),
        TierDecision(
            tier=AnalysisTier.AUDIT,
            name="Audit",
            status=TierStatus.EXECUTE,
            reason="Valid.",
        ),
        TierDecision(
            tier=AnalysisTier.REPAIR,
            name="Repair",
            status=TierStatus.EXECUTE,
            reason="Valid.",
        ),
        TierDecision(
            tier=AnalysisTier.INTELLIGENCE,
            name="Intelligence",
            status=TierStatus.EXECUTE,
            reason="Valid.",
        ),
        TierDecision(
            tier=AnalysisTier.INTELLIGENCE,
            name="Engineering",
            status=TierStatus.EXECUTE,
            reason="Duplicate tier.",
        ),
    ]

    try:
        ExecutionPlan(
            selected_tiers=[
                1,
                2,
                3,
                4,
                5,
                6,
            ],
            tier_decisions=invalid_decisions,
            detected_domain="general",
        )
    except ValueError as error:
        assert (
            "duplicate tiers"
            in str(error)
            or "tiers 1 through 6"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected duplicate tiers to be rejected."
        )


def test_invalid_sql_is_rejected() -> None:
    try:
        create_execution_plan(
            primary_sql=(
                "DELETE FROM dataset"
            )
        )
    except ValueError as error:
        assert "SELECT" in str(error)
    else:
        raise AssertionError(
            "Expected non-SELECT SQL to be rejected."
        )


def main() -> None:
    print("=" * 60)
    print("NEXUS-6 EXECUTION PLANNER TEST")
    print("=" * 60)

    test_default_tier_decisions()
    print("PASS: Default six-tier decisions")

    test_tier_decision_model()
    print("PASS: Tier decision model")

    test_execution_plan_contains_all_six_tiers()
    print("PASS: Six-tier execution plan")

    test_execution_plan_json_serialization()
    print("PASS: JSON serialization")

    test_execution_plan_round_trip()
    print("PASS: JSON round-trip validation")

    test_invalid_tier_decisions_are_rejected()
    print("PASS: Invalid tier-count protection")

    test_invalid_tier_number_is_rejected()
    print("PASS: Duplicate-tier protection")

    test_invalid_sql_is_rejected()
    print("PASS: SQL validation")

    print()
    print("=" * 60)
    print("ALL EXECUTION PLANNER TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()