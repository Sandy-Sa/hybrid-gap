from pydantic import BaseModel, Field

from coding.models.extraction import Diagnosis, Evidence, Procedure


class SpecificityUpgrade(BaseModel):
    """A note-documented concept that supports a more specific code than the
    one already assigned. Emitted only by the vllm gap-filling prompt."""

    name: str
    normalized: str
    current_code: str
    evidence: list[Evidence]


class GapAnalysisResult(BaseModel):
    """Schema for gap analysis — allows empty arrays (no gaps found).

    `specificity_upgrades` is optional and defaults to an empty list; only the
    vllm gap-filling prompt populates it.
    """

    diagnoses: list[Diagnosis] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)
    specificity_upgrades: list[SpecificityUpgrade] = Field(default_factory=list)
