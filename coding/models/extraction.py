from pydantic import BaseModel, Field


class Evidence(BaseModel):
    quote: str
    section: str


class Diagnosis(BaseModel):
    name: str
    normalized: str
    evidence: list[Evidence]


class Procedure(BaseModel):
    name: str
    normalized: str
    evidence: list[Evidence]


class ClinicalNoteExtraction(BaseModel):
    diagnoses: list[Diagnosis] = Field(min_length=1)
    procedures: list[Procedure]
