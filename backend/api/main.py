from pathlib import Path

from fastapi import FastAPI, HTTPException

from backend.api.schemas import AnalysisRequest
from backend.core.evidence import DatasetEvidenceBuilder
from backend.core.intelligence import DatasetIntelligence
from backend.core.loader import DataEngine
from backend.core.observation_collector import ObservationCollector
from backend.core.orchestrator import (
    NexusOrchestrator,
    OrchestrationRequest,
)
from backend.core.quality import DataQualityAnalyzer
from backend.core.relationships import RelationshipAnalyzer
from backend.core.statistics import StatisticalAnalyzer


app = FastAPI(
    title="NEXUS-6",
    description="Universal Agentic Data Orchestrator & Migration Engine",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "nexus-6",
    }


@app.post("/analyze")
def analyze(request: AnalysisRequest) -> dict[str, object]:
    dataset_path = Path(request.dataset)

    if not dataset_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Dataset file does not exist: {request.dataset}",
        )

    if not dataset_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Dataset path is not a file: {request.dataset}",
        )

    engine = DataEngine()

    try:
        engine.read_file(
            str(dataset_path),
            table_name="dataset",
        )

        intelligence = DatasetIntelligence(
            engine=engine,
            table_name="dataset",
        )

        quality_analyzer = DataQualityAnalyzer(
            engine=engine,
            table_name="dataset",
        )

        relationship_analyzer = RelationshipAnalyzer(
            engine=engine,
            table_name="dataset",
        )

        statistical_analyzer = StatisticalAnalyzer(
            engine=engine,
            table_name="dataset",
        )

        observation_collector = ObservationCollector(
            quality_analyzer=quality_analyzer,
            relationship_analyzer=relationship_analyzer,
            statistical_analyzer=statistical_analyzer,
        )

        evidence_builder = DatasetEvidenceBuilder(
            intelligence=intelligence,
            observation_collector=observation_collector,
        )

        evidence = evidence_builder.build()

        orchestration_request = OrchestrationRequest(
            objective=request.objective,
            context=request.context,
        )

        orchestrator = NexusOrchestrator()

        execution_plan = orchestrator.orchestrate(
            evidence=evidence,
            request=orchestration_request,
        )

        return {
            "status": "success",
            "dataset": request.dataset,
            "evidence": evidence.to_dict(),
            "execution_plan": execution_plan.model_dump(
                mode="json"
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    finally:
        engine.close()