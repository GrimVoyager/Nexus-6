from backend.core.critic import CriticStatus, CriticValidator

from backend.core.evidence import DatasetEvidenceBuilder

from backend.core.evidence_merger import EvidenceMerger

from backend.core.executor import TierExecutor

from backend.core.intelligence import DatasetIntelligence

from backend.core.loader import DataEngine

from backend.core.observation_collector import ObservationCollector

from backend.core.orchestrator import (

    NexusOrchestrator,

    OrchestrationRequest,

)

from backend.core.quality import DataQualityAnalyzer

from backend.core.relationships import RelationshipAnalyzer

from backend.core.replanner import Replanner, ReplanState

from backend.core.statistics import StatisticalAnalyzer

from pathlib import Path
from copy import copy

from tempfile import TemporaryDirectory







CSV_CONTENT = """Employee_ID,Department,Years_Experience,Salary,Performance_Score

E001,Engineering,2,50000,72

E002,Engineering,4,65000,81

E003,HR,3,55000,76

E004,Sales,5,70000,88

E005,Sales,7,85000,91

E006,Finance,6,80000,87

E007,Finance,8,95000,94

E008,Engineering,10,120000,96

E009,HR,1,42000,68

E010,Sales,4,62000,79

E011,Finance,9,105000,93

E012,Engineering,6,90000,89

"""





def create_dataset():

    temporary_directory = TemporaryDirectory()



    dataset_path = (

        Path(temporary_directory.name)

        / "employees.csv"

    )



    dataset_path.write_text(

        CSV_CONTENT,

        encoding="utf-8",

    )



    engine = DataEngine()



    engine.read_file(

        dataset_path,

        table_name="employees",

    )



    return temporary_directory, engine





def build_components(engine: DataEngine):

    intelligence_engine = DatasetIntelligence(

        engine=engine,

        table_name="employees",

    )



    quality_analyzer = DataQualityAnalyzer(

        engine=engine,

        table_name="employees",

    )



    relationship_analyzer = RelationshipAnalyzer(

        engine=engine,

        table_name="employees",

    )



    statistical_analyzer = StatisticalAnalyzer(

        engine=engine,

        table_name="employees",

    )



    observation_collector = ObservationCollector(

        quality_analyzer=quality_analyzer,

        relationship_analyzer=relationship_analyzer,

        statistical_analyzer=statistical_analyzer,

    )



    evidence_builder = DatasetEvidenceBuilder(

        intelligence=intelligence_engine,

        observation_collector=observation_collector,

    )



    evidence = evidence_builder.build()



    executor = TierExecutor(
        intelligence=intelligence_engine,
        quality_analyzer=quality_analyzer,
        relationship_analyzer=relationship_analyzer,
        statistical_analyzer=statistical_analyzer,
        observation_collector=observation_collector,
    )



    return evidence, executor





def create_query_objective() -> str:

    return (

        "Run a SQL query to retrieve "

        "employees with salary above 80000."

    )





def remove_query_execution(execution_result):

    """

    Create an intentionally incomplete execution result.



    This simulates a case where the orchestrator requested

    Tier 1 but the executor did not actually complete it.



    The source dataset is never modified.

    """



    incomplete_result = copy(execution_result)

    incomplete_result.executed_tiers = [
        tier
        for tier in execution_result.executed_tiers
        if tier != 1
    ]

    incomplete_result.execution_details = {
        key: value
        for key, value in (
            execution_result.execution_details
            or {}
        ).items()
        if str(key) != "1"
    }

    return incomplete_result





def test_sufficient_evidence_stops_replanning():

    temporary_directory, engine = create_dataset()



    try:

        evidence, executor = build_components(engine)



        objective = (

            "Analyze employee relationships "

            "and identify important patterns."

        )



        orchestrator = NexusOrchestrator()



        plan = orchestrator.orchestrate(

            evidence=evidence,

            request=OrchestrationRequest(

                objective=objective,

            ),

        )



        execution_result = executor.execute(plan=plan, evidence=evidence)



        critic = CriticValidator()



        decision = critic.evaluate(

            objective=objective,

            evidence=evidence,

            execution_result=execution_result,

        )



        assert decision.status == CriticStatus.SUFFICIENT

        assert decision.replan_required is False



        print(

            "PASS: Sufficient evidence stops replanning"

        )



    finally:

        engine.close()

        temporary_directory.cleanup()





def test_missing_query_execution_triggers_replan():

    temporary_directory, engine = create_dataset()



    try:

        evidence, executor = build_components(engine)



        objective = create_query_objective()



        orchestrator = NexusOrchestrator()



        initial_plan = orchestrator.orchestrate(

            evidence=evidence,

            request=OrchestrationRequest(

                objective=objective,

            ),

        )



        initial_query_decision = next(

            decision

            for decision in initial_plan.tier_decisions

            if int(decision.tier) == 1

        )



        assert initial_query_decision.status == "EXECUTE"



        initial_execution_result = executor.execute(
            plan=initial_plan,
            evidence=evidence,
        )



        assert 1 in initial_execution_result.executed_tiers



        incomplete_execution_result = (

            remove_query_execution(

                initial_execution_result

            )

        )



        critic = CriticValidator()



        decision = critic.evaluate(

            objective=objective,

            evidence=evidence,

            execution_result=incomplete_execution_result,

        )



        assert decision.status == CriticStatus.INSUFFICIENT

        assert decision.replan_required is True

        assert (

            "query_execution"

            in decision.missing_requirements

        )



        replanner = Replanner()



        replan_decision, revised_plan = replanner.replan(

            current_plan=initial_plan,

            critic_decision=decision,

            replan_count=0,

        )



        assert (

            replan_decision.state

            == ReplanState.REPLANNING

        )



        assert replan_decision.replan_required is True



        assert 1 in replan_decision.requested_tiers



        revised_query_decision = next(

            decision

            for decision in revised_plan.tier_decisions

            if int(decision.tier) == 1

        )



        assert revised_query_decision.status == "EXECUTE"



        revised_execution_result = executor.execute(
            plan=revised_plan,
            evidence=evidence,
        )



        assert (

            1

            in revised_execution_result.executed_tiers

        )



        print(

            "PASS: Missing query execution triggers replan"

        )



    finally:

        engine.close()

        temporary_directory.cleanup()





def test_revised_plan_reaches_executor():

    temporary_directory, engine = create_dataset()



    try:

        evidence, executor = build_components(engine)



        objective = create_query_objective()



        orchestrator = NexusOrchestrator()



        initial_plan = orchestrator.orchestrate(

            evidence=evidence,

            request=OrchestrationRequest(

                objective=objective,

            ),

        )



        initial_execution_result = executor.execute(
            plan=initial_plan,
            evidence=evidence,
        )



        incomplete_execution_result = (

            remove_query_execution(

                initial_execution_result

            )

        )



        critic = CriticValidator()



        critic_decision = critic.evaluate(

            objective=objective,

            evidence=evidence,

            execution_result=incomplete_execution_result,

        )



        assert (

            critic_decision.status

            == CriticStatus.INSUFFICIENT

        )



        replanner = Replanner()



        replan_decision, revised_plan = replanner.replan(

            current_plan=initial_plan,

            critic_decision=critic_decision,

            replan_count=0,

        )



        assert (

            replan_decision.state

            == ReplanState.REPLANNING

        )



        revised_execution_result = executor.execute(
            plan=revised_plan,
            evidence=evidence,
        )



        assert (

            1

            in revised_execution_result.executed_tiers

        )



        print(

            "PASS: Revised plan reaches executor"

        )



    finally:

        engine.close()

        temporary_directory.cleanup()





def test_revised_plan_preserves_six_tier_structure():

    temporary_directory, engine = create_dataset()



    try:

        evidence, executor = build_components(engine)



        objective = create_query_objective()



        orchestrator = NexusOrchestrator()



        initial_plan = orchestrator.orchestrate(

            evidence=evidence,

            request=OrchestrationRequest(

                objective=objective,

            ),

        )



        execution_result = executor.execute(
            plan=initial_plan,
            evidence=evidence,
        )



        incomplete_execution_result = (

            remove_query_execution(

                execution_result

            )

        )



        critic = CriticValidator()



        critic_decision = critic.evaluate(

            objective=objective,

            evidence=evidence,

            execution_result=incomplete_execution_result,

        )



        replanner = Replanner()



        _, revised_plan = replanner.replan(

            current_plan=initial_plan,

            critic_decision=critic_decision,

            replan_count=0,

        )



        assert revised_plan.selected_tiers == [

            1,

            2,

            3,

            4,

            5,

            6,

        ]



        assert len(revised_plan.tier_decisions) == 6



        assert sorted(

            int(decision.tier)

            for decision in revised_plan.tier_decisions

        ) == [

            1,

            2,

            3,

            4,

            5,

            6,

        ]



        print(

            "PASS: Revised plan preserves six-tier structure"

        )



    finally:

        engine.close()

        temporary_directory.cleanup()





def test_replan_count_protects_agentic_loop():

    temporary_directory, engine = create_dataset()



    try:

        evidence, executor = build_components(engine)



        objective = create_query_objective()



        orchestrator = NexusOrchestrator()



        plan = orchestrator.orchestrate(

            evidence=evidence,

            request=OrchestrationRequest(

                objective=objective,

            ),

        )



        execution_result = executor.execute(plan=plan, evidence=evidence)



        incomplete_execution_result = (

            remove_query_execution(

                execution_result

            )

        )



        critic = CriticValidator()



        critic_decision = critic.evaluate(

            objective=objective,

            evidence=evidence,

            execution_result=incomplete_execution_result,

        )



        replanner = Replanner(

            max_replans=3,

        )



        replan_decision, revised_plan = replanner.replan(

            current_plan=plan,

            critic_decision=critic_decision,

            replan_count=3,

        )



        assert (

            replan_decision.state

            == ReplanState.FAILED

        )



        assert replan_decision.replan_required is False



        
        assert revised_plan is None



        print(

            "PASS: Replan count protects agentic loop"

        )



    finally:

        engine.close()

        temporary_directory.cleanup()





if __name__ == "__main__":

    print("=" * 60)

    print("NEXUS-6 AGENTIC REPLAN PIPELINE TEST")

    print("=" * 60)



    test_sufficient_evidence_stops_replanning()

    test_missing_query_execution_triggers_replan()

    test_revised_plan_reaches_executor()

    test_revised_plan_preserves_six_tier_structure()

    test_replan_count_protects_agentic_loop()



    print("=" * 60)

    print(

        "ALL NEXUS-6 AGENTIC REPLAN PIPELINE TESTS PASSED"

    )

    print("=" * 60)