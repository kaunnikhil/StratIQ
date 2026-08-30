"""
Ambiguity Layer: if no hypothesis clears the confidence threshold, don't
force a single answer — surface the shortlist, name the evidence gaps, and
specify a concrete next step to resolve it.
"""
from typing import List

from backend.models import AmbiguityResult, EvidenceBundle, ScoredHypothesis

CONFIDENCE_THRESHOLD = 0.60


def _evidence_gap(h: ScoredHypothesis) -> str:
    if h.evidence_strength < 0.4:
        return f'"{h.statement}" has weak supporting evidence — needs a more direct data source, not just adjacent signals.'
    if h.temporal_alignment < 0.4:
        return f'"{h.statement}" is not clearly established as preceding the anomaly — timing needs to be confirmed.'
    if h.comparative_analysis < 0.4:
        return f'"{h.statement}" is not well-differentiated from the control region — unclear if this is local or market-wide.'
    return f'"{h.statement}" has moderate support but is not decisively ahead of alternatives.'


def _next_action(h: ScoredHypothesis, bundle: EvidenceBundle) -> str:
    text = h.statement.lower()
    if "competitor" in text or "rival" in text:
        return "Pull competitor pricing/scheme data at the branch level and compare loan-decline reasons cited by customers."
    if "flood" in text or "footfall" in text:
        return "Pull daily (not weekly) footfall data for the flood-affected branches to confirm the recovery timeline matches the disbursal drop."
    if "agent" in text or "staff" in text:
        return "Compare disbursal trends specifically in the two attrition-affected branches versus the other four Pune branches."
    if "cibil" in text or "credit score" in text:
        return "Check whether the national CIBIL cutoff change shows a comparable rejection-rate increase in the control region."
    if "repo" in text or "rate" in text or "emi" in text:
        return "Check whether the control region shows a comparable disbursal dip after the same repo rate change."
    return "Escalate to the relevant branch manager or dealer partner for direct, on-the-ground confirmation."


def assess_ambiguity(scored: List[ScoredHypothesis], bundle: EvidenceBundle,
                      threshold: float = CONFIDENCE_THRESHOLD) -> AmbiguityResult:
    if not scored:
        top_confidence = 0.0
    else:
        top_confidence = scored[0].confidence

    is_ambiguous = top_confidence < threshold
    # Only surface gaps/next actions for the ambiguous case, or for the top 3
    # candidates when confident, to keep the output focused either way.
    relevant = scored if is_ambiguous else scored[:1]
    candidates_for_gaps = scored[:3] if is_ambiguous else []

    gaps = [_evidence_gap(h) for h in candidates_for_gaps]
    actions = [_next_action(h, bundle) for h in (candidates_for_gaps or relevant)]

    return AmbiguityResult(
        is_ambiguous=is_ambiguous,
        threshold=threshold,
        top_confidence=top_confidence,
        ranked_hypotheses=scored,
        evidence_gaps=gaps,
        next_actions=actions,
    )
