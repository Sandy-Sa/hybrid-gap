from pydantic import BaseModel


class CodeVerdict(BaseModel):
    code: str
    evidence_quote: str
    contradiction_quote: str | None = None
    decision: str  # "Confirmed" or "Rule out"
    correct: bool | None = None


class VerificationResponse(BaseModel):
    verdicts: list[CodeVerdict]
