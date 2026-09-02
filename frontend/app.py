import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import pandas as pd
import streamlit as st

from backend.signal_engine import detect_anomaly, SignalEngineError
from backend.vector_store import EvidenceVectorStore
from backend.evidence_fusion import fuse_evidence
from backend.llm_client import get_llm_client, GeminiHypothesisLLM
from backend.hypothesis_engine import run_hypothesis_engine
from backend.ambiguity_layer import assess_ambiguity
from backend.narrative import build_narrative
import data.scenario_config as cfg

st.set_page_config(page_title="StratIQ", layout="wide")
st.title("StratIQ")
st.caption("An AI analyst, not an AI narrator : StratIQ detects meaningful changes, connects structured and unstructured evidence, ranks competing explanations, and surfaces what to investigate next")

SCENARIOS = {
    "Suvidha Finserv — Pune (Core)": {
        "target_branches": cfg.PUNE_BRANCHES,
        "control_branches": cfg.NASHIK_BRANCHES,
        "target_region": cfg.TARGET_REGION,
        "control_region": cfg.CONTROL_REGION,
        "title": "Two-Wheeler Loan Disbursals — Pune vs. Nashik",
        "description": "Round 1 scenario: investigate the Pune disbursal decline.",
        "sparse": False,
    },
    "Aurangabad — Naturally Ambiguous": {
        "target_branches": cfg.AMBIGUOUS_TARGET_BRANCHES,
        "control_branches": cfg.AMBIGUOUS_CONTROL_BRANCHES,
        "target_region": cfg.AMBIGUOUS_TARGET_REGION,
        "control_region": cfg.AMBIGUOUS_CONTROL_REGION,
        "title": "Two-Wheeler Loan Disbursals — Ambiguous Pilot",
        "description": "Natural low-confidence scenario: several plausible drivers, no dominant explanation.",
        "sparse": False,
    },
    "Pune VimanNagar — Sparse History": {
        "target_branches": [cfg.SPARSE_BRANCH],
        "control_branches": cfg.NASHIK_BRANCHES,
        "target_region": cfg.SPARSE_REGION,
        "control_region": cfg.CONTROL_REGION,
        "title": "Two-Wheeler Loan Disbursals — New Pilot Branch",
        "description": "Sparse-history scenario: only three weeks of KPI history.",
        "sparse": True,
    },
}

selected_scenario = st.selectbox(
    "Investigation scenario",
    list(SCENARIOS.keys()),
)

scenario = SCENARIOS[selected_scenario]
st.caption(scenario["description"])

@st.cache_resource
def get_store():
    store = EvidenceVectorStore(persist_dir="data/index")
    store.index_json_sources(cfg.UNSTRUCTURED_DIR)
    return store


def load_series(regions):
    conn = sqlite3.connect(cfg.DB_PATH)
    placeholders = ",".join("?" for _ in regions)
    df = pd.read_sql_query(
        f"""
        SELECT region, week, SUM(value) AS value
        FROM disbursals
        WHERE region IN ({placeholders})
        GROUP BY region, week
        ORDER BY region, week
        """,
        conn,
        params=regions,
    )
    conn.close()
    return df.pivot(index="week", columns="region", values="value")


df = load_series([
    scenario["target_region"],
    scenario["control_region"],
])

st.subheader(scenario["title"])
st.line_chart(df)

llm = get_llm_client()
provider_note = "Live Gemini reasoning" if isinstance(llm, GeminiHypothesisLLM) else "Offline heuristic fallback (no GEMINI_API_KEY set — set one in .env for full reasoning quality)"
st.caption(f"LLM provider: {provider_note}")

if st.button("Investigate scenario", type="primary"):
    with st.spinner("Running signal check, evidence fusion, and hypothesis ranking..."):
        try:
            signal = detect_anomaly(
                cfg.DB_PATH,
                scenario["target_branches"],
                scenario["control_branches"],
                target_label=scenario["target_region"],
                control_label=scenario["control_region"],
            )
        except SignalEngineError as e:
            if scenario["sparse"]:
                st.info(
                    "Insufficient history — StratIQ abstains from anomaly detection "
                    "because this KPI has only 3 weeks of observations. "
                    "At least 6 consecutive weeks are required."
                )
            else:
                st.error(str(e))
            st.stop()

        store = get_store()
        bundle = fuse_evidence(
                            cfg.DB_PATH,
                            signal,
                            scenario["target_branches"],
                            scenario["control_branches"],
                            store,
                            region=scenario["target_region"],
                            top_k=10,
                        )
        scored, telemetry = run_hypothesis_engine(llm, bundle)
        ambiguity = assess_ambiguity(scored, bundle)
        narrative = build_narrative(signal, ambiguity)
        evidence_by_id = {d.doc_id: d for d in bundle.unstructured_evidence}

    st.divider()
    badge = "🔴 Genuine anomaly detected" if signal.is_anomaly else "🟢 Normal variance — no investigation warranted"
    st.subheader(badge)
    c1, c2, c3 = st.columns(3)
    c1.metric("Pune change", f"{signal.change_pct:+.1f}%")
    c2.metric("Nashik (control) change", f"{signal.control_change_pct:+.1f}%")
    c3.metric("Anomaly window", f"Week {signal.anomaly_start_week}–{signal.anomaly_end_week}")

    st.divider()
    st.subheader("Processing Breakdown — LLM vs. Non-LLM")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Signal detection", "Non-LLM", help="ruptures change-point detection + control comparison, pure statistics")
    p2.metric("Evidence retrieval", "Non-LLM", help="SQL structured lookup + TF-IDF cosine similarity")
    p3.metric("Hypothesis generation", "LLM" if telemetry and telemetry.is_llm_call else "Non-LLM (fallback)",
              help="Gemini proposes candidate causes from the fused evidence")
    p4.metric("Confidence scoring", "Non-LLM", help="Evidence strength, temporal alignment, comparative analysis — all computed in Python, never asserted by the LLM")

    if telemetry:
        st.caption("Runtime telemetry (this run)")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Latency", f"{telemetry.latency_seconds:.2f}s")
        t2.metric("Model calls", "1" if telemetry.is_llm_call else "0")
        t3.metric("Total tokens", f"{telemetry.total_tokens:,}" if telemetry.is_llm_call else "n/a")
        t4.metric("Est. cost", f"${telemetry.estimated_cost_usd:.5f}" if telemetry.is_llm_call else "$0.00000")
        if telemetry.is_llm_call:
            st.caption(f"Provider: {telemetry.provider} · Model: {telemetry.model} · Prompt tokens: {telemetry.prompt_tokens:,} · Output tokens: {telemetry.output_tokens:,} (pricing is an approximate published-rate estimate, not a live billing figure)")
        else:
            st.caption(f"Provider: {telemetry.provider} ({telemetry.model}) — deterministic, no API call made, zero cost")

    st.divider()
    if ambiguity.is_ambiguous:
        st.warning(f"Ambiguous — top confidence {ambiguity.top_confidence:.0%} is below the {ambiguity.threshold:.0%} threshold. No single cause is being forced.")
    else:
        st.success(f"Leading explanation identified — confidence {ambiguity.top_confidence:.0%}")

    st.subheader("Ranked Hypotheses")
    for h in ambiguity.ranked_hypotheses:
        with st.expander(f"#{h.rank} · {h.confidence:.0%} confidence · {h.statement}"):
            st.write(h.mechanism)
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Evidence strength", f"{h.evidence_strength:.0%}")
            cc2.metric("Temporal alignment", f"{h.temporal_alignment:.0%}")
            cc3.metric("Comparative analysis", f"{h.comparative_analysis:.0%}")
            st.write("**Supporting evidence:**")
            if h.supporting_evidence_ids:
                for eid in h.supporting_evidence_ids:
                    doc = evidence_by_id.get(eid)
                    if doc:
                        freshness = signal.anomaly_start_week - doc.week if doc.week is not None else None
                        freshness_str = f"{freshness}wk before anomaly start" if freshness is not None and freshness >= 0 else \
                                        (f"{abs(freshness)}wk after anomaly start" if freshness is not None else "week unknown")
                        st.caption(f"`{eid}` · source: {doc.source} · method: {doc.method} · relevance: {doc.relevance:.2f} · {freshness_str}")
                    else:
                        st.caption(f"`{eid}` · (structured evidence, see source tables)")
            else:
                st.write("none")
            if h.contradictory_evidence_ids:
                st.write("**Contradictory evidence:**", ", ".join(h.contradictory_evidence_ids))

    if ambiguity.is_ambiguous:
        st.subheader("Evidence Gaps")
        for g in ambiguity.evidence_gaps:
            st.write("- " + g)
        st.subheader("Next Actions to Resolve")
        for a in ambiguity.next_actions:
            st.write("- " + a)

    st.divider()
    st.subheader("Narrative Summary")
    st.code(narrative, language=None)
