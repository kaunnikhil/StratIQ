"""
Narrative layer: converts the ambiguity-aware result into the
Claim -> Evidence -> Confidence -> Evidence Gap -> Next Action format.
"""
from backend.models import AmbiguityResult, SignalResult


def build_narrative(signal: SignalResult, ambiguity: AmbiguityResult) -> str:
    lines = []
    lines.append(
        f"SIGNAL: {signal.target_label} disbursals changed {signal.change_pct:+.1f}% "
        f"(week {signal.anomaly_start_week}-{signal.anomaly_end_week}) versus "
        f"{signal.control_change_pct:+.1f}% for {signal.control_label} — "
        f"{'a genuine, localized deviation' if signal.is_anomaly else 'within normal variance, no investigation warranted'}."
    )

    if not ambiguity.ranked_hypotheses:
        lines.append("No hypotheses could be generated from the available evidence.")
        return "\n".join(lines)

    if not ambiguity.is_ambiguous:
        top = ambiguity.ranked_hypotheses[0]
        lines.append(f"CLAIM: {top.statement}")
        lines.append(f"EVIDENCE: {top.mechanism} (supported by {len(top.supporting_evidence_ids)} evidence items)")
        lines.append(
            f"CONFIDENCE: {top.confidence:.0%} "
            f"(evidence strength {top.evidence_strength:.0%}, temporal alignment {top.temporal_alignment:.0%}, "
            f"comparative analysis {top.comparative_analysis:.0%})"
        )
        lines.append("RECOMMENDED ACTION: " + (ambiguity.next_actions[0] if ambiguity.next_actions else "Review with the relevant business owner before acting."))
    else:
        lines.append(
            f"AMBIGUOUS: top hypothesis confidence ({ambiguity.top_confidence:.0%}) is below the "
            f"{ambiguity.threshold:.0%} threshold — no single cause is forced. Ranked competing explanations:"
        )
        for h in ambiguity.ranked_hypotheses[:3]:
            lines.append(f"  [{h.confidence:.0%}] {h.statement}")
        lines.append("EVIDENCE GAPS:")
        for g in ambiguity.evidence_gaps:
            lines.append(f"  - {g}")
        lines.append("NEXT ACTIONS TO RESOLVE:")
        for a in ambiguity.next_actions:
            lines.append(f"  - {a}")

    return "\n".join(lines)
