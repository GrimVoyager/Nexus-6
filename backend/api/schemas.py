from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    dataset: str = Field(
        min_length=1,
        description="Dataset reference or dataset name.",
    )

    context: str = Field(
        min_length=1,
        description="Business or analytical context.",
    )

    objective: str = Field(
        min_length=1,
        description="What the user wants NEXUS-6 to accomplish.",
    )
