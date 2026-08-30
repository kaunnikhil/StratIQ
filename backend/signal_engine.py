"""
Signal Engine: distinguishes a genuine KPI anomaly from normal noise using
change-point detection, with a control series (e.g. a comparable but
unaffected region) used as a market-wide sanity check, not as the target's
own baseline.
"""
import sqlite3
from typing import List

import numpy as np
import pandas as pd
import ruptures as rpt

from backend.models import SignalResult


class SignalEngineError(Exception):
    pass


def _weekly_series(db_path: str, branches: List[str]) -> pd.DataFrame:
    if not branches:
        raise SignalEngineError("No branches provided for series lookup.")
    placeholders = ",".join("?" for _ in branches)
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            f"SELECT week, SUM(value) as value FROM disbursals "
            f"WHERE branch IN ({placeholders}) GROUP BY week ORDER BY week",
            conn, params=branches,
        )
    finally:
        conn.close()
    if df.empty:
        raise SignalEngineError(f"No disbursal data found for branches: {branches}")
    weeks = df["week"].tolist()
    if weeks != list(range(min(weeks), max(weeks) + 1)):
        raise SignalEngineError("Weekly series is not consecutive; check source data.")
    if len(df) < 6:
        raise SignalEngineError("Insufficient history to run change-point detection (need >= 6 weeks).")
    return df


def detect_anomaly(
    db_path: str,
    target_branches: List[str],
    control_branches: List[str],
    target_label: str = "target",
    control_label: str = "control",
    min_pct_change: float = 0.08,
) -> SignalResult:
    """
    Detects a structural change point in the target series and evaluates it
    against the control series to decide whether it's a genuine, localized
    anomaly or part of a broader (market-wide) movement / normal noise.
    """
    target_df = _weekly_series(db_path, target_branches)
    control_df = _weekly_series(db_path, control_branches)

    values = target_df["value"].to_numpy()
    algo = rpt.Binseg(model="l2").fit(values)
    # Ask for 1 breakpoint (plus the trailing end-of-series index ruptures always returns)
    result = algo.predict(n_bkps=1)
    change_idx = result[0]  # index into the array where the new regime starts

    weeks = target_df["week"].to_numpy()
    if change_idx >= len(weeks):
        change_idx = len(weeks) - 1
    anomaly_start_week = int(weeks[change_idx])
    anomaly_end_week = int(weeks[-1])

    # Change-point detectors can land 1-2 weeks late relative to the true
    # trigger. Exclude a 1-week buffer immediately before the detected start
    # from the baseline average, so a week that's already shifting doesn't
    # dilute the "normal" baseline and understate the true change magnitude.
    baseline_cutoff = anomaly_start_week - 1
    baseline_mask = target_df["week"] < baseline_cutoff
    anomaly_mask = target_df["week"] >= anomaly_start_week
    if baseline_mask.sum() == 0:
        # Not enough history before the buffer; fall back to the unbuffered split.
        baseline_mask = target_df["week"] < anomaly_start_week
    if baseline_mask.sum() == 0 or anomaly_mask.sum() == 0:
        raise SignalEngineError("Change point detection did not produce a valid split.")

    baseline_value = float(target_df.loc[baseline_mask, "value"].mean())
    anomaly_value = float(target_df.loc[anomaly_mask, "value"].mean())
    change_pct = (anomaly_value - baseline_value) / baseline_value * 100

    # Control comparison over the same window
    c_baseline_mask = control_df["week"] < anomaly_start_week
    c_anomaly_mask = control_df["week"] >= anomaly_start_week
    control_baseline = float(control_df.loc[c_baseline_mask, "value"].mean())
    control_anomaly = float(control_df.loc[c_anomaly_mask, "value"].mean())
    control_change_pct = (control_anomaly - control_baseline) / control_baseline * 100 if control_baseline else 0.0

    is_genuine = abs(change_pct / 100) >= min_pct_change and \
        abs(change_pct - control_change_pct) >= (min_pct_change * 100 * 0.5)

    # Confidence: larger, more control-divergent moves get higher confidence
    magnitude_score = min(abs(change_pct) / 25.0, 1.0)
    divergence_score = min(abs(change_pct - control_change_pct) / 25.0, 1.0)
    confidence = round(0.5 * magnitude_score + 0.5 * divergence_score, 2)

    return SignalResult(
        is_anomaly=bool(is_genuine),
        anomaly_start_week=anomaly_start_week,
        anomaly_end_week=anomaly_end_week,
        baseline_value=round(baseline_value, 2),
        anomaly_value=round(anomaly_value, 2),
        change_pct=round(change_pct, 2),
        control_change_pct=round(control_change_pct, 2),
        confidence=confidence,
        target_label=target_label,
        control_label=control_label,
    )
