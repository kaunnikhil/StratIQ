# StratIQ

**An AI analyst, not an AI narrator.** CausalBoard explains why a business KPI moved, ranks how confident it is in each explanation, and recommends the next action — instead of just visualizing that the metric changed.

Built for the Accenture Innovation Challenge 2026, BusinessIntelligence.ai problem statement: *"A dashboard can show revenue dropped 8% in a region; it rarely explains why or what to do next."*

## Problem

Explaining a KPI anomaly today takes 2–3 days of manual cross-referencing across disconnected systems (sales, ops, HR, support tickets, news). CausalBoard is built directly against the problem statement's three hardest asks:

1. **Signal vs. noise** — is this a genuine anomaly, or normal variance?
2. **Correlation → action** — not just "what moved together," but a specific, confidence-scored, actionable explanation.
3. **Genuine ambiguity** — when multiple causes are plausible at once, don't force a false-confidence answer; say what would resolve it.

## Demo scenario

**Suvidha Finserv** (illustrative NBFC): two-wheeler loan disbursals in the Pune cluster fall ~18% over four weeks, while a comparable control cluster (Nashik) and the national trend stay flat. Five plausible causes co-occur: a competitor's zero-down-payment scheme, monsoon flooding, field-agent attrition, a national CIBIL cutoff tightening, and an RBI repo-rate hike. The synthetic dataset's baseline growth is grounded in real published FY25 NBFC two-wheeler financing figures (~11% YoY national growth, ~68.5% NBFC market share, ₹70,000–90,000 average ticket size) — only the Pune-specific anomaly and its causes are synthetic.

## Architecture

```
Signal Engine  →  Evidence Fusion  →  Hypothesis Engine  →  Ambiguity Layer  →  Narrative
(change-point     (structured SQL      (LLM proposes,        (confidence         (Claim → Evidence →
 detection vs.      + TF-IDF retrieval   Python scores)        threshold check,    Confidence →
 control series)     of unstructured                           evidence gaps,      Evidence Gap →
                     evidence)                                 next actions)       Next Action)
```

1. **Signal Engine** (`backend/signal_engine.py`) — `ruptures` change-point detection on the target's weekly time series, cross-checked against a control region so a genuine localized anomaly can be distinguished from market-wide movement or noise.
2. **Evidence Fusion** (`backend/evidence_fusion.py`, `backend/vector_store.py`) — pulls all structured SQLite rows in the anomaly window (disbursals, agents, footfall, policy, rate, competitor events) and retrieves the most relevant unstructured documents (dealer notes, tickets, news, social) via TF-IDF + cosine similarity.
3. **Hypothesis Engine** (`backend/hypothesis_engine.py`, `backend/llm_client.py`) — an LLM (Gemini) proposes 3–5 competing hypotheses from the fused evidence; **all confidence scoring is computed in Python**, not asserted by the LLM, across three components: evidence strength (retrieval relevance), temporal alignment (does the cause precede the anomaly?), and comparative analysis (does the control region show the same pattern?).
4. **Ambiguity Layer** (`backend/ambiguity_layer.py`) — if no hypothesis clears a confidence threshold, the system does not force a conclusion. It surfaces the ranked shortlist, names the specific evidence gap per candidate, and recommends the exact next data cut or person to consult.
5. **Narrative** (`backend/narrative.py`) — renders the final result as Claim → Evidence → Confidence → Evidence Gap → Next Action.

### A deliberate engineering tradeoff, stated plainly
The retrieval layer uses **TF-IDF + cosine similarity** (scikit-learn) rather than dense embeddings (sentence-transformers/Chroma). Under a hard deadline, the dense-embedding stack hit a Windows OpenMP/PyTorch DLL conflict with no guaranteed fix in the available time. TF-IDF has no native-dependency risk, is fully sufficient for keyword/entity-rich business text like ours, and sits behind the same `search()` interface — swapping in dense embeddings later is a drop-in change, not a rewrite.

### Reliability fallback
`backend/llm_client.py` is provider-agnostic. If `GEMINI_API_KEY` is missing or invalid, the system automatically falls back to a deterministic offline heuristic generator, so the pipeline stays runnable (e.g., during a live demo without network access). **For real reasoning quality and hypothesis differentiation, a valid Gemini API key should be set** — the offline mode is a safety net, not the intended reasoning path.

## Tech stack

| Layer | Technology |
|---|---|
| Backend API | Python, FastAPI |
| Data | SQLite, pandas |
| Signal detection | ruptures |
| Retrieval | scikit-learn (TF-IDF + cosine similarity) |
| Reasoning | Google Gemini (`google-genai`), provider-agnostic client |
| Frontend / demo | Streamlit |
| Testing | pytest |

## Dependencies

See `requirements.txt`. Python 3.10+ recommended.

## Setup & execution

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GEMINI_API_KEY=<your key>   (optional — offline fallback works without it)

# 1. Generate the synthetic dataset
python data/generate_synthetic_data.py

# 2. Run the test suite
python -m pytest tests/ -q

# 3a. Run the API
uvicorn backend.main:app --reload
# POST http://127.0.0.1:8000/investigate  {"region": "Pune"}

# 3b. Or run the interactive demo UI
streamlit run frontend/app.py
```

## Repository structure

```
causalboard/
├── data/
│   ├── scenario_config.py       # scenario parameters (swap this to build a new scenario)
│   ├── generate_synthetic_data.py
│   ├── suvidha.db                # generated, not committed
│   ├── unstructured/              # generated, not committed
│   └── ground_truth/              # evaluation-only labels, not used by the pipeline
├── backend/
│   ├── models.py
│   ├── signal_engine.py
│   ├── vector_store.py
│   ├── evidence_fusion.py
│   ├── llm_client.py
│   ├── hypothesis_engine.py
│   ├── ambiguity_layer.py
│   ├── narrative.py
│   └── main.py
├── frontend/app.py
├── tests/test_pipeline.py
└── requirements.txt
```

## Validated results (Suvidha Finserv scenario)

- Signal Engine correctly flags Pune as a genuine, localized anomaly (control region change stays within normal range) — verified in `tests/test_pipeline.py`.
- Evidence Fusion returns supporting evidence for all five injected candidate causes.
- Hypothesis Engine produces a ranked, confidence-scored shortlist rather than a single assertion.
- Ambiguity Layer correctly triggers and produces evidence gaps + next actions when confidence is (deliberately, for testing) held to an artificially high threshold.

## Roadmap (beyond this prototype)

- **Universal Data Adapter** — LLM-based schema inference so any business's CSV/Excel + text data can be mapped into CausalBoard's internal representation without code changes, proving the "same architecture, different signals" scalability claim.
- **Reasoning Framework Library** — retrieval-augmented business methodologies (5 Whys, Kepner-Tregoe Problem Analysis, MECE issue trees, Porter's Five Forces) so the Hypothesis Engine adapts its reasoning approach to the type of anomaly, rather than reasoning generically.
- Dense-embedding retrieval as a drop-in upgrade once the platform dependency issue is resolved outside the deadline.
