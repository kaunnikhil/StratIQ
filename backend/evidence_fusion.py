"""
Evidence Fusion: given a SignalResult, pulls all relevant structured rows
within the anomaly window and retrieves the most relevant unstructured
documents, returning one combined EvidenceBundle for the Hypothesis Engine.
"""
import sqlite3
from typing import List, Optional

from backend.models import EvidenceBundle, SignalResult, StructuredEvidenceRow
from backend.vector_store import EvidenceVectorStore


def _rows_as_dicts(conn, query, params=()):
    conn.row_factory = sqlite3.Row
    cur = conn.execute(query, params)
    return [dict(r) for r in cur.fetchall()]


def fuse_evidence(
    db_path: str,
    signal: SignalResult,
    target_branches: List[str],
    control_branches: List[str],
    store: EvidenceVectorStore,
    region: Optional[str] = None,
    top_k: int = 12,
) -> EvidenceBundle:
    conn = sqlite3.connect(db_path)
    structured: List[StructuredEvidenceRow] = []
    counts = {}

    # Widen the window slightly: change-point detection can land a week or two
    # off from the true trigger, and root-cause events often precede the
    # detected shift, so evidence lookup should not be pinned exactly to it.
    w0_raw, w1 = signal.anomaly_start_week, signal.anomaly_end_week
    w0 = max(1, w0_raw - 2)
    target_ph = ",".join("?" for _ in target_branches)
    control_ph = ",".join("?" for _ in control_branches)

    disb_target = _rows_as_dicts(
        conn,
        f"SELECT * FROM disbursals WHERE branch IN ({target_ph}) AND week BETWEEN ? AND ?",
        (*target_branches, w0, w1),
    )
    disb_control = _rows_as_dicts(
        conn,
        f"SELECT * FROM disbursals WHERE branch IN ({control_ph}) AND week BETWEEN ? AND ?",
        (*control_branches, w0, w1),
    )
    agents = _rows_as_dicts(
        conn, f"SELECT * FROM agents WHERE branch IN ({target_ph}) AND week BETWEEN ? AND ?",
        (*target_branches, w0, w1),
    )
    footfall = _rows_as_dicts(
        conn, f"SELECT * FROM footfall WHERE branch IN ({target_ph}) AND week BETWEEN ? AND ?",
        (*target_branches, w0, w1),
    )
    cibil = _rows_as_dicts(conn, "SELECT * FROM cibil_policy WHERE effective_week <= ?", (w1,))
    repo = _rows_as_dicts(conn, "SELECT * FROM repo_rate WHERE effective_week <= ?", (w1,))
    competitor_q = "SELECT * FROM competitor_events WHERE week BETWEEN ? AND ?"
    competitor_params = (w0, w1)
    if region:
        competitor_q += " AND region = ?"
        competitor_params = (w0, w1, region)
    competitor = _rows_as_dicts(conn, competitor_q, competitor_params)

    conn.close()

    for source_name, rows in [
        ("disbursals_target", disb_target), ("disbursals_control", disb_control),
        ("agents", agents), ("footfall", footfall),
        ("cibil_policy", cibil), ("repo_rate", repo), ("competitor_events", competitor),
    ]:
        counts[source_name] = len(rows)
        for row in rows:
            structured.append(StructuredEvidenceRow(source=source_name, data=row))

    # Build a retrieval query from the anomaly context
    query = f"disbursal decline {region or ''} branch loans agent staffing competitor scheme flooding credit score interest rate week {w0} to {w1}"
    unstructured = store.search(query, top_k=top_k, week_min=max(1, w0 - 2), week_max=w1, region=region)
    counts["unstructured_documents"] = len(unstructured)

    return EvidenceBundle(
        signal=signal,
        structured_evidence=structured,
        unstructured_evidence=unstructured,
        source_record_counts=counts,
    )
