from typing import List, Optional
from pydantic import BaseModel


class SignalResult(BaseModel):
    is_anomaly: bool
    anomaly_start_week: int
    anomaly_end_week: int
    baseline_value: float
    anomaly_value: float
    change_pct: float
    control_change_pct: float
    confidence: float
    target_label: str
    control_label: str


class StructuredEvidenceRow(BaseModel):
    source: str
    data: dict
    method: str = "sql_structured_lookup"


class UnstructuredDoc(BaseModel):
    doc_id: str
    source: str
    week: Optional[int] = None
    region: Optional[str] = None
    branch: Optional[str] = None
    text: str
    relevance: float = 0.0
    method: str = "tfidf_retrieval"


class LLMTelemetry(BaseModel):
    provider: str
    model: str
    latency_seconds: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    is_llm_call: bool = True


class EvidenceBundle(BaseModel):
    signal: SignalResult
    structured_evidence: List[StructuredEvidenceRow]
    unstructured_evidence: List[UnstructuredDoc]
    source_record_counts: dict


class Hypothesis(BaseModel):
    statement: str
    mechanism: str
    supporting_evidence_ids: List[str]
    contradictory_evidence_ids: List[str] = []
    relevant_weeks: List[int] = []


class ScoredHypothesis(BaseModel):
    statement: str
    mechanism: str
    supporting_evidence_ids: List[str]
    contradictory_evidence_ids: List[str]
    evidence_strength: float
    temporal_alignment: float
    comparative_analysis: float
    confidence: float
    rank: int


class AmbiguityResult(BaseModel):
    is_ambiguous: bool
    threshold: float
    top_confidence: float
    ranked_hypotheses: List[ScoredHypothesis]
    evidence_gaps: List[str]
    next_actions: List[str]


class InvestigationResult(BaseModel):
    signal: SignalResult
    ambiguity: AmbiguityResult
    narrative: str
    telemetry: Optional[LLMTelemetry] = None
