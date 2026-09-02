from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.signal_engine import detect_anomaly, SignalEngineError
from backend.vector_store import EvidenceVectorStore
from backend.evidence_fusion import fuse_evidence
from backend.llm_client import get_llm_client
from backend.hypothesis_engine import run_hypothesis_engine
from backend.ambiguity_layer import assess_ambiguity
from backend.narrative import build_narrative
from backend.models import InvestigationResult
import data.scenario_config as cfg

app = FastAPI(title="CausalBoard API")

_store = None


def _get_store():
    global _store
    if _store is None:
        _store = EvidenceVectorStore(persist_dir="data/index")
        _store.index_json_sources(cfg.UNSTRUCTURED_DIR)
    return _store


class InvestigateRequest(BaseModel):
    region: str = "Pune"


@app.post("/investigate", response_model=InvestigationResult)
def investigate(req: InvestigateRequest):
    try:
        signal = detect_anomaly(
            cfg.DB_PATH, cfg.PUNE_BRANCHES, cfg.NASHIK_BRANCHES,
            target_label=cfg.TARGET_REGION, control_label=cfg.CONTROL_REGION,
        )
    except SignalEngineError as e:
        raise HTTPException(status_code=422, detail=str(e))

    store = _get_store()
    bundle = fuse_evidence(
        cfg.DB_PATH, signal, cfg.PUNE_BRANCHES, cfg.NASHIK_BRANCHES,
        store, region=req.region, top_k=10,
    )
    llm = get_llm_client()
    scored, telemetry = run_hypothesis_engine(llm, bundle)
    ambiguity = assess_ambiguity(scored, bundle)
    narrative = build_narrative(signal, ambiguity)

    return InvestigationResult(signal=signal, ambiguity=ambiguity, narrative=narrative, telemetry=telemetry)


@app.get("/health")
def health():
    return {"status": "ok"}
