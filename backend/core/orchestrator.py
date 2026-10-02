from __future__ import annotations

import re
from typing import Any

from backend.core.evidence import DatasetEvidencePackage
from backend.core.planner import (
    AnalysisTier,
    DomainType,
    ExecutionPlan,
    TierDecision,
    TierStatus,
)


class OrchestrationRequest:
    """
    Input contract for the deterministic NEXUS-6 orchestrator.

    The request contains the user's analytical objective and optional
    context. Dataset facts come from DatasetEvidencePackage.
    """

    def __init__(
        self,
        objective: str,
        context: str = "",
    ) -> None:
        normalized_objective = objective.strip()

        if not normalized_objective:
            raise ValueError(
                "objective must not be empty"
            )

        self.objective = normalized_objective
        self.context = context.strip()


class NexusOrchestrator:
    """
    Deterministic baseline orchestrator for NEXUS-6.

    Responsibilities:
        - Interpret the basic shape of the user's objective.
        - Detect a likely domain from dataset metadata.
        - Evaluate dataset capabilities.
        - Decide how each universal tier should be handled.
        - Identify a possible supervised-learning target state.
        - Produce a validated ExecutionPlan.

    This version intentionally does not use an LLM.

    The orchestrator decides WHAT should happen.
    Analytical engines decide HOW to calculate it.
    """

    DOMAIN_KEYWORDS: dict[DomainType, tuple[str, ...]] = {
        "hr_workforce": (
            "employee",
            "employees",
            "employee_id",
            "department",
            "designation",
            "job_role",
            "job_title",
            "salary",
            "performance",
            "experience",
            "attrition",
            "tenure",
            "hire_date",
            "joining_date",
            "job_satisfaction",
        ),
        "bfsi_finance": (
            "loan",
            "loan_amount",
            "loan_status",
            "credit",
            "credit_score",
            "interest_rate",
            "income",
            "balance",
            "account",
            "account_number",
            "transaction",
            "transaction_amount",
            "default",
            "fraud",
            "insurance",
            "premium",
        ),
        "healthcare": (
            "patient",
            "patient_id",
            "diagnosis",
            "disease",
            "symptom",
            "blood_pressure",
            "glucose",
            "heart_rate",
            "hospital",
            "admission",
            "discharge",
            "medical",
            "treatment",
            "medication",
        ),
        "retail_ecom": (
            "product",
            "product_id",
            "product_name",
            "order",
            "order_id",
            "customer",
            "customer_id",
            "price",
            "quantity",
            "discount",
            "revenue",
            "sales",
            "cart",
            "category",
            "sku",
        ),
    }

    QUERY_TERMS = (
        "query",
        "sql",
        "show me",
        "find",
        "filter",
        "rows where",
        "records where",
        "how many",
        "count",
        "calculate",
    )

    ENGINEERING_TERMS = (
        "migration",
        "migrate",
        "schema migration",
        "ddl",
        "dml",
        "database",
        "dbt",
        "pyspark",
        "spark",
        "etl",
        "pipeline",
        "warehouse",
        "transform",
        "convert",
    )

    REPAIR_TERMS = (
        "clean",
        "cleanup",
        "repair",
        "fix",
        "missing",
        "duplicate",
        "duplicates",
        "impute",
        "invalid",
        "quality",
    )

    DEEP_ANALYSIS_TERMS = (
        "deep analysis",
        "deep dive",
        "analyze",
        "analyse",
        "analysis",
        "insights",
        "relationships",
        "correlation",
        "patterns",
        "trend",
        "trends",
        "predict",
        "prediction",
        "model",
        "machine learning",
        "ml",
        "forecast",
    )

    SNAPSHOT_TERMS = (
        "overview",
        "summary",
        "snapshot",
        "profile",
        "describe",
        "describe the data",
    )

    TARGET_TERMS = (
        "target",
        "predict",
        "prediction",
        "predicting",
        "outcome",
        "label",
        "dependent variable",
    )

    def orchestrate(
        self,
        *,
        evidence: DatasetEvidencePackage,
        request: OrchestrationRequest,
    ) -> ExecutionPlan:
        objective = request.objective.lower()

        domain, domain_score = self._detect_domain(
            evidence
        )

        objective_type = self._detect_objective_type(
            objective
        )

        capabilities = evidence.intelligence.get(
            "capabilities",
            {},
        )

        target_state, target = self._detect_target(
            evidence=evidence,
            objective=objective,
        )

        tier_decisions = self._build_tier_decisions(
            evidence=evidence,
            objective=objective,
            objective_type=objective_type,
            capabilities=capabilities,
            target_state=target_state,
        )

        overlay = (
            domain != "general"
            and domain_score >= 2
        )

        reasoning = self._build_reasoning(
            domain=domain,
            domain_score=domain_score,
            objective_type=objective_type,
            target_state=target_state,
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
            tier_decisions=tier_decisions,
            detected_domain=domain,
            overlay=overlay,
            target=target,
            reasoning=reasoning,
        )

    def _detect_domain(
        self,
        evidence: DatasetEvidencePackage,
    ) -> tuple[DomainType, int]:
        """
        Detect a domain using dataset column names.

        A domain is selected only when one domain has at least two
        matching signals and scores strictly higher than every other
        domain.
        """

        columns = evidence.intelligence.get(
            "columns",
            [],
        )

        column_names = [
            str(column.get("name", "")).lower()
            for column in columns
        ]

        scores: dict[DomainType, int] = {
            "hr_workforce": 0,
            "bfsi_finance": 0,
            "healthcare": 0,
            "retail_ecom": 0,
            "general": 0,
        }

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for column_name in column_names:
                normalized_column = re.sub(
                    r"[^a-z0-9]+",
                    "_",
                    column_name,
                ).strip("_")

                if (
                    normalized_column in keywords
                ):
                    scores[domain] += 1
                    continue

                if any(
                    keyword in normalized_column
                    for keyword in keywords
                ):
                    scores[domain] += 1

        ranked_domains = sorted(
            (
                (domain, score)
                for domain, score in scores.items()
                if domain != "general"
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        if not ranked_domains:
            return "general", 0

        first_domain, first_score = (
            ranked_domains[0]
        )

        second_score = (
            ranked_domains[1][1]
            if len(ranked_domains) > 1
            else 0
        )

        if first_score < 2:
            return "general", first_score

        if first_score == second_score:
            return "general", first_score

        return first_domain, first_score

    def _detect_objective_type(
        self,
        objective: str,
    ) -> str:
        if self._contains_any(
            objective,
            self.ENGINEERING_TERMS,
        ):
            return "engineering"

        if self._contains_any(
            objective,
            self.QUERY_TERMS,
        ):
            return "query"

        if self._contains_any(
            objective,
            self.REPAIR_TERMS,
        ):
            return "repair"

        if self._contains_any(
            objective,
            self.DEEP_ANALYSIS_TERMS,
        ):
            return "deep_analysis"

        if self._contains_any(
            objective,
            self.SNAPSHOT_TERMS,
        ):
            return "snapshot"

        return "general"

    def _detect_target(
        self,
        *,
        evidence: DatasetEvidencePackage,
        objective: str,
    ) -> tuple[str, str | None]:
        potential_targets = evidence.intelligence.get(
            "potential_targets",
            [],
        )

        if not potential_targets:
            return "NO_TARGET", None

        if not self._contains_any(
            objective,
            self.TARGET_TERMS,
        ):
            return "TARGET_UNCERTAIN", None

        objective_tokens = set(
            re.findall(
                r"[a-zA-Z_][a-zA-Z0-9_]*",
                objective.lower(),
            )
        )

        exact_matches: list[dict[str, Any]] = []

        for candidate in potential_targets:
            column_name = str(
                candidate.get("column", "")
            )

            normalized_column = column_name.lower()

            if normalized_column in objective_tokens:
                exact_matches.append(candidate)
                continue

            if normalized_column.replace(
                "_",
                " ",
            ) in objective:
                exact_matches.append(candidate)

        if len(exact_matches) == 1:
            return (
                "TARGET_FOUND",
                str(
                    exact_matches[0]["column"]
                ),
            )

        return "TARGET_UNCERTAIN", None

    def _build_tier_decisions(
        self,
        *,
        evidence: DatasetEvidencePackage,
        objective: str,
        objective_type: str,
        capabilities: dict[str, Any],
        target_state: str,
    ) -> list[TierDecision]:
        row_count = evidence.row_count
        column_count = evidence.column_count

        decisions: list[TierDecision] = []

        # Tier 1: Query
        query_capable = bool(
            capabilities.get("query", False)
        )

        if (
            objective_type == "query"
            and query_capable
        ):
            query_status = TierStatus.EXECUTE
            query_reason = (
                "The objective contains a query-oriented request "
                "and the dataset supports analytical querying."
            )
        elif objective_type == "query":
            query_status = TierStatus.PARTIAL
            query_reason = (
                "The objective requests a query, but the dataset "
                "does not currently expose sufficient query capability."
            )
        else:
            query_status = TierStatus.NOT_APPLICABLE
            query_reason = (
                "The current objective does not require a "
                "query-specific execution."
            )

        decisions.append(
            TierDecision(
                tier=AnalysisTier.QUERY,
                name="Query",
                status=query_status,
                reason=query_reason,
            )
        )

        # Tier 2: Overview
        overview_capable = bool(
            capabilities.get("overview", False)
        )

        if overview_capable:
            decisions.append(
                TierDecision(
                    tier=AnalysisTier.OVERVIEW,
                    name="Overview",
                    status=TierStatus.EXECUTE,
                    reason=(
                        "The dataset contains usable rows and can "
                        "support deterministic overview analysis."
                    ),
                )
            )
        else:
            decisions.append(
                TierDecision(
                    tier=AnalysisTier.OVERVIEW,
                    name="Overview",
                    status=TierStatus.NOT_APPLICABLE,
                    reason=(
                        "The dataset does not contain sufficient "
                        "data for overview analysis."
                    ),
                )
            )

        # Tier 3: Audit
        audit_capable = bool(
            capabilities.get("audit", False)
        )

        if audit_capable:
            decisions.append(
                TierDecision(
                    tier=AnalysisTier.AUDIT,
                    name="Audit",
                    status=TierStatus.EXECUTE,
                    reason=(
                        "Dataset structure is available for deterministic "
                        "quality and statistical auditing."
                    ),
                )
            )
        else:
            decisions.append(
                TierDecision(
                    tier=AnalysisTier.AUDIT,
                    name="Audit",
                    status=TierStatus.NOT_APPLICABLE,
                    reason=(
                        "The dataset does not expose enough structure "
                        "for an audit."
                    ),
                )
            )

        # Tier 4: Repair
        quality_summary = evidence.intelligence.get(
            "quality_summary",
            {},
        )

        quality_issues = (
            evidence.summary.type_counts.get(
                "missing_values",
                0,
            )
            > 0
            or evidence.summary.type_counts.get(
                "duplicate_rows",
                0,
            )
            > 0
            or evidence.summary.type_counts.get(
                "numeric_validity",
                0,
            )
            > 0
        )

        if (
            objective_type == "repair"
            and quality_issues
        ):
            repair_status = TierStatus.PARTIAL
            repair_reason = (
                "Repair-related evidence exists, but this tier only "
                "plans repair work. It does not silently mutate source data."
            )
        elif objective_type == "repair":
            repair_status = TierStatus.PARTIAL
            repair_reason = (
                "A repair objective was requested. The repair tier "
                "will require further planning before any mutation."
            )
        elif quality_issues:
            repair_status = TierStatus.PARTIAL
            repair_reason = (
                "Quality evidence indicates possible repair work, "
                "but no repair mutation is authorized by the planner."
            )
        else:
            repair_status = TierStatus.NOT_APPLICABLE
            repair_reason = (
                "No repair-specific evidence or repair objective "
                "requires this tier."
            )

        if quality_summary:
            repair_reason += (
                " Deterministic quality summary is available."
            )

        decisions.append(
            TierDecision(
                tier=AnalysisTier.REPAIR,
                name="Repair",
                status=repair_status,
                reason=repair_reason,
            )
        )

        # Tier 5: Intelligence
        intelligence_capable = (
            bool(
                capabilities.get(
                    "relationships",
                    False,
                )
            )
            or bool(
                capabilities.get(
                    "predictive_modeling",
                    False,
                )
            )
            or bool(
                capabilities.get(
                    "categorical_analysis",
                    False,
                )
            )
        )

        if (
            objective_type == "deep_analysis"
            and intelligence_capable
        ):
            intelligence_status = TierStatus.EXECUTE
            intelligence_reason = (
                "The objective requests deeper analysis and the dataset "
                "contains capabilities suitable for analytical intelligence."
            )
        elif intelligence_capable:
            intelligence_status = TierStatus.EXECUTE
            intelligence_reason = (
                "The dataset supports analytical intelligence through "
                "its available relationships, categorical structure, "
                "or predictive capability."
            )
        elif (
            target_state == "TARGET_FOUND"
            or target_state == "TARGET_UNCERTAIN"
        ):
            intelligence_status = TierStatus.PARTIAL
            intelligence_reason = (
                "A possible analytical target exists, but the dataset "
                "does not currently expose sufficient intelligence capability."
            )
        else:
            intelligence_status = TierStatus.NOT_APPLICABLE
            intelligence_reason = (
                "The dataset does not currently expose sufficient "
                "capability for the intelligence tier."
            )

        decisions.append(
            TierDecision(
                tier=AnalysisTier.INTELLIGENCE,
                name="Intelligence",
                status=intelligence_status,
                reason=intelligence_reason,
            )
        )

        # Tier 6: Engineering
        engineering_capable = bool(
            capabilities.get(
                "engineering_analysis",
                False,
            )
        )

        if (
            objective_type == "engineering"
            and engineering_capable
        ):
            engineering_status = TierStatus.EXECUTE
            engineering_reason = (
                "The objective explicitly requests engineering or "
                "migration analysis and the dataset exposes engineering capability."
            )
        elif objective_type == "engineering":
            engineering_status = TierStatus.PARTIAL
            engineering_reason = (
                "An engineering objective was requested, but the "
                "current dataset does not expose sufficient engineering capability."
            )
        else:
            engineering_status = TierStatus.NOT_APPLICABLE
            engineering_reason = (
                "No engineering, migration, ETL, or schema-conversion "
                "objective has been detected."
            )

        decisions.append(
            TierDecision(
                tier=AnalysisTier.ENGINEERING,
                name="Engineering",
                status=engineering_status,
                reason=engineering_reason,
            )
        )

        return decisions

    @staticmethod
    def _contains_any(
        text: str,
        terms: tuple[str, ...],
    ) -> bool:
        return any(
            term in text
            for term in terms
        )

    @staticmethod
    def _build_reasoning(
        *,
        domain: DomainType,
        domain_score: int,
        objective_type: str,
        target_state: str,
    ) -> str:
        return (
            f"Detected domain '{domain}' with "
            f"{domain_score} matching dataset signals. "
            f"Objective type is '{objective_type}'. "
            f"Target state is '{target_state}'."
        )