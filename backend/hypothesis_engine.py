"""
Hypothesis Engine: the LLM proposes candidate explanations; Python computes
every score. This keeps confidence numbers auditable and prevents the LLM
from asserting its own certainty.
"""
from typing import List

from backend.llm_client import HypothesisLLM
from backend.models import EvidenceBundle, Hypothesis, ScoredHypothesis

WEIGHTS = {"evidence_strength": 0.45, "temporal_alignment": 0.30, "comparative_analysis": 0.25}


def _evidence_strength(hyp: Hypothesis, bundle: EvidenceBundle) -> float:
    """Based on retrieval relevance of the supporting unstructured docs and
    whether structured evidence exists for the same theme."""
    relevance_by_id = {d.doc_id: d.relevance for d in bundle.unstructured_evidence}
    if not hyp.supporting_evidence_ids:
        return 0.1
    scores = [relevance_by_id.get(eid, 0.0) for eid in hyp.supporting_evidence_ids]
    scores = [s for s in scores if s > 0]
    if not scores:
        return 0.2
    avg_relevance = sum(scores) / len(scores)
    coverage_bonus = min(len(scores) / 4, 1.0) * 0.2
    return round(min(avg_relevance * 3 + coverage_bonus, 1.0), 3)


def _temporal_alignment(hyp: Hypothesis, bundle: EvidenceBundle) -> float:
    """Rewards causes whose evidence weeks precede or coincide with the
    anomaly start; penalizes causes whose only evidence comes after it."""
    start = bundle.signal.anomaly_start_week
    if not hyp.relevant_weeks:
        return 0.4
    diffs = [start - w for w in hyp.relevant_weeks]  # positive = precedes anomaly (good)
    best = max(diffs)
    if best < 0:
        return 0.15  # evidence only appears after the anomaly began -> weak causal case
    if best == 0:
        return 0.75
    return round(min(0.75 + best * 0.08, 1.0), 3)


def _comparative_analysis(hyp: Hypothesis, bundle: EvidenceBundle) -> float:
    """National/market-wide causes should score LOWER here when the control
    region (e.g. Nashik) did not move, since that argues against a
    market-wide explanation and for a localized one."""
    control_moved = abs(bundle.signal.control_change_pct) >= 5.0
    text = (hyp.statement + " " + hyp.mechanism).lower()
    is_national_cause = any(k in text for k in ["repo", "cibil", "national", "rate hike", "credit score"])
    if is_national_cause:
        return 0.75 if control_moved else 0.35
    else:
        return 0.35 if control_moved else 0.85


def score_hypotheses(hypotheses: List[Hypothesis], bundle: EvidenceBundle) -> List[ScoredHypothesis]:
    scored = []
    for h in hypotheses:
        es = _evidence_strength(h, bundle)
        ta = _temporal_alignment(h, bundle)
        ca = _comparative_analysis(h, bundle)
        confidence = round(
            es * WEIGHTS["evidence_strength"] +
            ta * WEIGHTS["temporal_alignment"] +
            ca * WEIGHTS["comparative_analysis"], 3
        )
        scored.append(ScoredHypothesis(
            statement=h.statement, mechanism=h.mechanism,
            supporting_evidence_ids=h.supporting_evidence_ids,
            contradictory_evidence_ids=h.contradictory_evidence_ids,
            evidence_strength=es, temporal_alignment=ta, comparative_analysis=ca,
            confidence=confidence, rank=0,
        ))
    scored.sort(key=lambda s: s.confidence, reverse=True)
    for i, s in enumerate(scored, start=1):
        s.rank = i
    return scored


def run_hypothesis_engine(llm: HypothesisLLM, bundle: EvidenceBundle) -> List[ScoredHypothesis]:
    raw_hypotheses = llm.generate_hypotheses(bundle)
    return score_hypotheses(raw_hypotheses, bundle)
