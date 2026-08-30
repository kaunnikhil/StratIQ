"""
Scenario configuration for the Suvidha Finserv synthetic dataset.
Designed so a different scenario (e.g. retail) can be created later by
changing this config, not by rewriting the generator/pipeline.
"""

SEED = 42

# 12 weeks of history; anomaly window is weeks 5-8 (1-indexed)
NUM_WEEKS = 12
ANOMALY_START_WEEK = 5
ANOMALY_END_WEEK = 8

TARGET_REGION = "Pune"
CONTROL_REGION = "Nashik"

PUNE_BRANCHES = [
    "Pune-Shivajinagar", "Pune-Kothrud", "Pune-Hadapsar",
    "Pune-Pimpri", "Pune-Kondhwa", "Pune-Wagholi",
]
NASHIK_BRANCHES = ["Nashik-College-Road", "Nashik-Gangapur", "Nashik-Panchavati"]

# Branches affected by monsoon flooding (footfall hit)
FLOOD_AFFECTED_BRANCHES = ["Pune-Hadapsar", "Pune-Kondhwa"]
# Branches that lost a field agent (sourcing capacity hit)
AGENT_ATTRITION_BRANCHES = ["Pune-Kothrud", "Pune-Wagholi"]

# Real-world grounded baseline figures (FY25 NBFC two-wheeler financing, India)
NATIONAL_YOY_GROWTH = 0.11          # ~11% YoY NBFC 2W disbursement growth
NBFC_MARKET_SHARE = 0.685           # ~68.5% NBFC share of 2W financing
AVG_TICKET_SIZE_LOW = 70_000
AVG_TICKET_SIZE_HIGH = 90_000

# Weekly base disbursal count per branch (before value multiplication)
BASE_LOANS_PER_BRANCH_PER_WEEK = 22

PUNE_ANOMALY_DROP_PCT = 0.18  # ~18% decline in Pune from week 5

# National CIBIL policy tightening, 2 weeks before the anomaly
CIBIL_POLICY_EFFECTIVE_WEEK = 3
CIBIL_OLD_CUTOFF = 650
CIBIL_NEW_CUTOFF = 700

# RBI repo rate change (national, week 4)
REPO_RATE_EFFECTIVE_WEEK = 4
REPO_RATE_OLD = 6.50
REPO_RATE_NEW = 6.75

# Competitor zero-down-payment scheme launch (Pune region only), week 5
COMPETITOR_EVENT_WEEK = 5
COMPETITOR_EVENT = "Rival NBFC launches zero-down-payment two-wheeler scheme in Pune"

DB_PATH = "data/suvidha.db"
UNSTRUCTURED_DIR = "data/unstructured"
GROUND_TRUTH_PATH = "data/ground_truth/suvidha_pune_labels.json"

# Anchor date for week 1 (used to timestamp unstructured docs)
TIME_ORIGIN = "2025-04-07"
