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


@st.cache_resource
def get_store():
    store = EvidenceVectorStore(persist_dir="data/index")
    store.index_json_sources(cfg.UNSTRUCTURED_DIR)
    return store


def load_series():
    conn = sqlite3.connect(cfg.DB_PATH)
    df = pd.read_sql_query(
        "SELECT region, week, SUM(value) as value FROM disbursals GROUP BY region, week ORDER BY region, week",
        conn,
    )
    conn.close()
    return df.pivot(index="week", columns="region", values="value")


df = load_series()
st.subheader("Two-Wheeler Loan Disbursals — Pune vs. Nashik (control) vs. National context")
st.line_chart(df)

llm = get_llm_client()
provider_note = "Live Gemini reasoning" if isinstance(llm, GeminiHypothesisLLM) else "Offline heuristic fallback (no GEMINI_API_KEY set — set one in .env for full reasoning quality)"
st.caption(f"LLM provider: {provider_note}")

if st.button("Investigate the Pune anomaly", type="primary"):
    with st.spinner("Running signal check, evidence fusion, and hypothesis ranking..."):
        try:
            signal = detect_anomaly(
                cfg.DB_PATH, cfg.PUNE_BRANCHES, cfg.NASHIK_BRANCHES,
                target_label="Pune", control_label="Nashik",
            )
        except SignalEngineError as e:
            st.error(str(e))
            st.stop()

        store = get_store()
        bundle = fuse_evidence(cfg.DB_PATH, signal, cfg.PUNE_BRANCHES, cfg.NASHIK_BRANCHES, store, region="Pune", top_k=10)
        scored = run_hypothesis_engine(llm, bundle)
        ambiguity = assess_ambiguity(scored, bundle)
        narrative = build_narrative(signal, ambiguity)

    st.divider()
    badge = "🔴 Genuine anomaly detected" if signal.is_anomaly else "🟢 Normal variance — no investigation warranted"
    st.subheader(badge)
    c1, c2, c3 = st.columns(3)
    c1.metric("Pune change", f"{signal.change_pct:+.1f}%")
    c2.metric("Nashik (control) change", f"{signal.control_change_pct:+.1f}%")
    c3.metric("Anomaly window", f"Week {signal.anomaly_start_week}–{signal.anomaly_end_week}")

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
            st.write("**Supporting evidence:**", ", ".join(h.supporting_evidence_ids) or "none")
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
