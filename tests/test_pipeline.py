import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.signal_engine import detect_anomaly
from backend.vector_store import EvidenceVectorStore
from backend.evidence_fusion import fuse_evidence
from backend.llm_client import OfflineHeuristicLLM
from backend.hypothesis_engine import run_hypothesis_engine
from backend.ambiguity_layer import assess_ambiguity
import data.scenario_config as cfg


def test_signal_engine_detects_pune_anomaly():
    signal = detect_anomaly(cfg.DB_PATH, cfg.PUNE_BRANCHES, cfg.NASHIK_BRANCHES, "Pune", "Nashik")
    assert signal.is_anomaly is True
    assert signal.change_pct < -8
    assert abs(signal.control_change_pct) < 8


def test_evidence_fusion_returns_all_sources():
    signal = detect_anomaly(cfg.DB_PATH, cfg.PUNE_BRANCHES, cfg.NASHIK_BRANCHES, "Pune", "Nashik")
    store = EvidenceVectorStore(persist_dir="data/index")
    store.index_json_sources(cfg.UNSTRUCTURED_DIR)
    bundle = fuse_evidence(cfg.DB_PATH, signal, cfg.PUNE_BRANCHES, cfg.NASHIK_BRANCHES, store, region="Pune", top_k=10)
    for source in ["disbursals_target", "agents", "footfall", "cibil_policy", "repo_rate", "competitor_events"]:
        assert bundle.source_record_counts.get(source, 0) > 0, f"missing evidence for {source}"


def test_offline_hypothesis_engine_produces_ranked_hypotheses():
    signal = detect_anomaly(cfg.DB_PATH, cfg.PUNE_BRANCHES, cfg.NASHIK_BRANCHES, "Pune", "Nashik")
    store = EvidenceVectorStore(persist_dir="data/index")
    store.index_json_sources(cfg.UNSTRUCTURED_DIR)
    bundle = fuse_evidence(cfg.DB_PATH, signal, cfg.PUNE_BRANCHES, cfg.NASHIK_BRANCHES, store, region="Pune", top_k=10)
    scored = run_hypothesis_engine(OfflineHeuristicLLM(), bundle)
    assert len(scored) >= 3
    assert scored[0].rank == 1
    assert scored[0].confidence >= scored[-1].confidence


def test_ambiguity_layer_triggers_below_threshold():
    signal = detect_anomaly(cfg.DB_PATH, cfg.PUNE_BRANCHES, cfg.NASHIK_BRANCHES, "Pune", "Nashik")
    store = EvidenceVectorStore(persist_dir="data/index")
    store.index_json_sources(cfg.UNSTRUCTURED_DIR)
    bundle = fuse_evidence(cfg.DB_PATH, signal, cfg.PUNE_BRANCHES, cfg.NASHIK_BRANCHES, store, region="Pune", top_k=10)
    scored = run_hypothesis_engine(OfflineHeuristicLLM(), bundle)
    result = assess_ambiguity(scored, bundle, threshold=0.99)  # force ambiguity for the test
    assert result.is_ambiguous is True
    assert len(result.evidence_gaps) > 0
    assert len(result.next_actions) > 0
