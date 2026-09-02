"""
Generates a reproducible, internally consistent synthetic dataset for the
Suvidha Finserv Pune-cluster scenario: a two-wheeler loan disbursal decline
with five plausible, co-occurring competing causes.

Run:
    python data/generate_synthetic_data.py
"""
import json
import os
import random
import sqlite3
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenario_config as cfg

random.seed(cfg.SEED)


def week_start_date(week: int) -> str:
    origin = datetime.strptime(cfg.TIME_ORIGIN, "%Y-%m-%d")
    return (origin + timedelta(weeks=week - 1)).strftime("%Y-%m-%d")


def build_tier2_scenarios(cur):
    """Add Tier 2 evaluation scenarios without changing the main Suvidha data."""
    rng = random.Random(cfg.SEED + 100)

    # ---------- Natural low-confidence scenario ----------
    # Real signal, but multiple weak/balanced drivers and a low-information
    # evidence layer. The existing 60% ambiguity threshold should trigger
    # naturally rather than being overridden.
    ambiguous_targets = [
        (b, cfg.AMBIGUOUS_TARGET_REGION)
        for b in cfg.AMBIGUOUS_TARGET_BRANCHES
    ]
    ambiguous_controls = [
        (b, cfg.AMBIGUOUS_CONTROL_REGION)
        for b in cfg.AMBIGUOUS_CONTROL_BRANCHES
    ]

    for branch, region in ambiguous_targets + ambiguous_controls:
        base_loans = 22 * rng.uniform(0.92, 1.08)
        for week in range(1, cfg.NUM_WEEKS + 1):
            trend = (1 + cfg.NATIONAL_YOY_GROWTH / 52) ** week
            noise = rng.uniform(0.97, 1.03)
            loan_count = base_loans * trend * noise

            if (
                region == cfg.AMBIGUOUS_TARGET_REGION
                and week >= cfg.AMBIGUOUS_ANOMALY_START_WEEK
            ):
                loan_count *= (1 - cfg.AMBIGUOUS_DROP_PCT) * rng.uniform(0.96, 1.04)

            ticket = (
                (cfg.AVG_TICKET_SIZE_LOW + cfg.AVG_TICKET_SIZE_HIGH) / 2
            ) * rng.uniform(0.94, 1.06)

            value = round(loan_count * ticket, 2)
            cur.execute(
                "INSERT INTO disbursals VALUES (?,?,?,?,?)",
                (branch, region, week, value, round(loan_count)),
            )

    # Balanced operational signals: no single driver dominates.
    for branch in cfg.AMBIGUOUS_TARGET_BRANCHES + cfg.AMBIGUOUS_CONTROL_BRANCHES:
        base_headcount = 6
        for week in range(1, cfg.NUM_WEEKS + 1):
            headcount = base_headcount
            if (
                branch == "Aurangabad-Waluj"
                and week >= 6
            ):
                headcount = 5
            cur.execute(
                "INSERT INTO agents VALUES (?,?,?)",
                (branch, week, headcount),
            )

    for branch in cfg.AMBIGUOUS_TARGET_BRANCHES + cfg.AMBIGUOUS_CONTROL_BRANCHES:
        base_footfall = 140
        for week in range(1, cfg.NUM_WEEKS + 1):
            count = base_footfall * rng.uniform(0.97, 1.03)

            if (
                branch in {"Aurangabad-Cidco", "Aurangabad-Garkheda"}
                and week >= cfg.AMBIGUOUS_ANOMALY_START_WEEK
            ):
                count *= 0.85

            cur.execute(
                "INSERT INTO footfall VALUES (?,?,?)",
                (branch, week, round(count)),
            )

    cur.execute(
        "INSERT INTO competitor_events VALUES (?,?,?)",
        (
            cfg.AMBIGUOUS_TARGET_REGION,
            cfg.AMBIGUOUS_ANOMALY_START_WEEK,
            "Rival lender launches a moderate promotional campaign in the region",
        ),
    )

    # ---------- Sparse-history scenario ----------
    # Exactly 3 weeks of disbursal history. The existing Signal Engine should
    # reject this cleanly because it requires >= 6 weeks.
    sparse_branch = cfg.SPARSE_BRANCH
    sparse_region = cfg.SPARSE_REGION

    base_loans = 22 * rng.uniform(0.95, 1.05)
    for week in range(1, cfg.SPARSE_WEEKS + 1):
        trend = (1 + cfg.NATIONAL_YOY_GROWTH / 52) ** week
        loan_count = base_loans * trend * rng.uniform(0.97, 1.03)
        ticket = (
            (cfg.AVG_TICKET_SIZE_LOW + cfg.AVG_TICKET_SIZE_HIGH) / 2
        ) * rng.uniform(0.95, 1.05)

        cur.execute(
            "INSERT INTO disbursals VALUES (?,?,?,?,?)",
            (
                sparse_branch,
                sparse_region,
                week,
                round(loan_count * ticket, 2),
                round(loan_count),
            ),
        )


def build_database():
    os.makedirs(os.path.dirname(cfg.DB_PATH), exist_ok=True)
    if os.path.exists(cfg.DB_PATH):
        os.remove(cfg.DB_PATH)
    conn = sqlite3.connect(cfg.DB_PATH)
    cur = conn.cursor()

    cur.execute("""CREATE TABLE disbursals (
        branch TEXT, region TEXT, week INTEGER, value REAL, loan_count INTEGER)""")
    cur.execute("""CREATE TABLE agents (
        branch TEXT, week INTEGER, headcount INTEGER)""")
    cur.execute("""CREATE TABLE footfall (
        branch TEXT, week INTEGER, count INTEGER)""")
    cur.execute("""CREATE TABLE cibil_policy (
        effective_week INTEGER, old_cutoff INTEGER, new_cutoff INTEGER)""")
    cur.execute("""CREATE TABLE repo_rate (
        effective_week INTEGER, rate REAL)""")
    cur.execute("""CREATE TABLE competitor_events (
        region TEXT, week INTEGER, event TEXT)""")

    all_branches = [(b, cfg.TARGET_REGION) for b in cfg.PUNE_BRANCHES] + \
                   [(b, cfg.CONTROL_REGION) for b in cfg.NASHIK_BRANCHES]

    avg_ticket = (cfg.AVG_TICKET_SIZE_LOW + cfg.AVG_TICKET_SIZE_HIGH) / 2
    weekly_growth = cfg.NATIONAL_YOY_GROWTH / 52  # simple weekly growth approximation

    for branch, region in all_branches:
        base_loans = cfg.BASE_LOANS_PER_BRANCH_PER_WEEK * random.uniform(0.9, 1.1)
        for week in range(1, cfg.NUM_WEEKS + 1):
            trend_factor = (1 + weekly_growth) ** week
            noise = random.uniform(0.95, 1.05)
            loan_count = base_loans * trend_factor * noise

            # Apply the Pune-only anomaly from the configured window
            if region == cfg.TARGET_REGION and cfg.ANOMALY_START_WEEK <= week <= cfg.NUM_WEEKS:
                loan_count *= (1 - cfg.PUNE_ANOMALY_DROP_PCT) * random.uniform(0.97, 1.03)

            ticket = avg_ticket * random.uniform(0.9, 1.1)
            value = round(loan_count * ticket, 2)
            cur.execute(
                "INSERT INTO disbursals VALUES (?,?,?,?,?)",
                (branch, region, week, value, round(loan_count)),
            )

    # Agents: attrition branches lose headcount from week 3
    for branch in cfg.PUNE_BRANCHES + cfg.NASHIK_BRANCHES:
        base_headcount = 6
        for week in range(1, cfg.NUM_WEEKS + 1):
            headcount = base_headcount
            if branch in cfg.AGENT_ATTRITION_BRANCHES and week >= 3:
                headcount = base_headcount - 2
            cur.execute("INSERT INTO agents VALUES (?,?,?)", (branch, week, headcount))

    # Footfall: flood-affected branches see reduced footfall from week 5
    for branch in cfg.PUNE_BRANCHES + cfg.NASHIK_BRANCHES:
        base_footfall = 140
        for week in range(1, cfg.NUM_WEEKS + 1):
            count = base_footfall * random.uniform(0.95, 1.05)
            if branch in cfg.FLOOD_AFFECTED_BRANCHES and cfg.ANOMALY_START_WEEK <= week <= cfg.ANOMALY_END_WEEK:
                count *= 0.6
            cur.execute("INSERT INTO footfall VALUES (?,?,?)", (branch, week, round(count)))

    cur.execute("INSERT INTO cibil_policy VALUES (?,?,?)",
                (cfg.CIBIL_POLICY_EFFECTIVE_WEEK, cfg.CIBIL_OLD_CUTOFF, cfg.CIBIL_NEW_CUTOFF))
    cur.execute("INSERT INTO repo_rate VALUES (?,?)",
                (cfg.REPO_RATE_EFFECTIVE_WEEK, cfg.REPO_RATE_NEW))
    cur.execute("INSERT INTO competitor_events VALUES (?,?,?)",
                (cfg.TARGET_REGION, cfg.COMPETITOR_EVENT_WEEK, cfg.COMPETITOR_EVENT))
    build_tier2_scenarios(cur)
    conn.commit()

    # Compute summary stats before closing
    def region_change(region):
        cur.execute("""SELECT week, SUM(value) FROM disbursals WHERE region=?
                       GROUP BY week ORDER BY week""", (region,))
        rows = cur.fetchall()
        baseline = sum(v for w, v in rows if w < cfg.ANOMALY_START_WEEK) / (cfg.ANOMALY_START_WEEK - 1)
        anomaly = sum(v for w, v in rows if cfg.ANOMALY_START_WEEK <= w <= cfg.ANOMALY_END_WEEK) / \
                  (cfg.ANOMALY_END_WEEK - cfg.ANOMALY_START_WEEK + 1)
        return baseline, anomaly, (anomaly - baseline) / baseline * 100

    pune_stats = region_change(cfg.TARGET_REGION)
    nashik_stats = region_change(cfg.CONTROL_REGION)

    conn.close()
    return pune_stats, nashik_stats


def build_unstructured():
    os.makedirs(cfg.UNSTRUCTURED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.GROUND_TRUTH_PATH), exist_ok=True)

    dealer_notes = [
        {"id": "DN-01", "week": 3, "branch": "Pune-Kothrud", "region": "Pune",
         "text": "Dealer mentioned one of our field executives has been unreachable this week; follow-up loan visits delayed."},
        {"id": "DN-02", "week": 4, "branch": "Pune-Wagholi", "region": "Pune",
         "text": "Sourcing agent resigned; showroom staff handling walk-ins directly for now, slower turnaround."},
        {"id": "DN-03", "week": 5, "branch": "Pune-Hadapsar", "region": "Pune",
         "text": "Heavy rain and waterlogging near the showroom this week, footfall visibly down."},
        {"id": "DN-04", "week": 5, "branch": "Pune-Kondhwa", "region": "Pune",
         "text": "Access road flooded for two days, customers postponed showroom visits."},
        {"id": "DN-05", "week": 5, "branch": "Pune-Shivajinagar", "region": "Pune",
         "text": "Customers asking if we can match the new zero-down-payment scheme from a rival NBFC."},
        {"id": "DN-06", "week": 6, "branch": "Pune-Pimpri", "region": "Pune",
         "text": "Walk-ins mentioning a competitor's no-down-payment offer advertised locally this month."},
        {"id": "DN-07", "week": 6, "branch": "Pune-Kothrud", "region": "Pune",
         "text": "Still short one field executive; backlog of pending loan file follow-ups growing."},
        {"id": "DN-08", "week": 4, "branch": "Pune-Shivajinagar", "region": "Pune",
         "text": "A few applicants rejected this week under the revised credit score cutoff."},
        {"id": "DN-09", "week": 7, "branch": "Pune-Hadapsar", "region": "Pune",
         "text": "Roads cleared but footfall still recovering slowly this week."},
        {"id": "DN-10", "week": 2, "branch": "Nashik-College-Road", "region": "Nashik",
         "text": "Steady week, no unusual footfall or staffing changes to report."},
        {"id": "DN-11", "week": 6, "branch": "Nashik-Gangapur", "region": "Nashik",
         "text": "Business as usual, in line with seasonal expectations."},
        {"id": "DN-12", "week": 4, "branch": "Pune-Wagholi", "region": "Pune",
         "text": "Interest rate queries increasing slightly after the repo rate news."},
        {"id": "DN-13", "week": 7, "branch": "Pune-Kondhwa", "region": "Pune",
         "text": "Local market still recovering from the flooding two weeks ago."},
        {"id": "DN-14", "week": 5, "branch": "Pune-Pimpri", "region": "Pune",
         "text": "No flooding or staffing issue here, but footfall dipped slightly this week too."},
        {"id": "DN-15", "week": 3, "branch": "Pune-Shivajinagar", "region": "Pune",
         "text": "New CIBIL cutoff means more applicants need a co-applicant to qualify."},
        {"id": "DN-16", "week": 8, "branch": "Pune-Kothrud", "region": "Pune",
         "text": "New field executive hired, expected to join next week."},
        {"id": "DN-17", "week": 6, "branch": "Pune-Hadapsar", "region": "Pune",
         "text": "Some customers went to the competitor showroom instead this week."},
        {"id": "DN-18", "week": 1, "branch": "Pune-Wagholi", "region": "Pune",
         "text": "Normal week, all targets on track."},
    ]

    tickets = [
        {"id": "TK-01", "week": 5, "region": "Pune",
         "text": "Customer complaint: was told a rival NBFC offers zero down payment, asked why we don't match it."},
        {"id": "TK-02", "week": 5, "region": "Pune",
         "text": "Customer unable to reach showroom due to flooded access road, requested reschedule."},
        {"id": "TK-03", "week": 4, "region": "Pune",
         "text": "Loan application rejected; customer asked why the credit score requirement increased."},
        {"id": "TK-04", "week": 6, "region": "Pune",
         "text": "Delay in follow-up call after initial enquiry, customer went ahead with a different lender."},
        {"id": "TK-05", "week": 7, "region": "Pune",
         "text": "Complaint about slow processing time attributed to reduced staff at branch."},
        {"id": "TK-06", "week": 3, "region": "Nashik",
         "text": "Minor query about EMI schedule, resolved same day."},
        {"id": "TK-07", "week": 4, "region": "Pune",
         "text": "Customer asked about the new interest rate after hearing about the RBI repo rate hike."},
        {"id": "TK-08", "week": 6, "region": "Pune",
         "text": "Customer mentioned competitor's advertisement offering no down payment this month."},
        {"id": "TK-09", "week": 2, "region": "Nashik",
         "text": "Standard document submission query."},
        {"id": "TK-10", "week": 7, "region": "Pune",
         "text": "Follow-up delayed again, customer frustrated with response time."},
    ]

    news = [
        {"id": "NW-01", "week": 5, "region": "Pune",
         "text": "Local news: heavy monsoon rainfall causes waterlogging in several Pune suburbs this week, disrupting local commerce."},
        {"id": "NW-02", "week": 4, "region": "national",
         "text": "RBI raises repo rate by 25 basis points, citing inflation concerns; lenders expected to pass on higher rates."},
        {"id": "NW-03", "week": 5, "region": "Pune",
         "text": "A rival two-wheeler financier announces a zero-down-payment scheme for new customers in the Pune region."},
    ]

    social = [
        {"id": "SM-01", "week": 5, "region": "Pune", "text": "Anyone know why [rival]'s zero down payment two-wheeler offer is so good right now? Thinking of switching."},
        {"id": "SM-02", "week": 5, "region": "Pune", "text": "Roads near Hadapsar completely waterlogged today, couldn't even get to the bike showroom."},
        {"id": "SM-03", "week": 6, "region": "Pune", "text": "Got rejected for a two-wheeler loan, apparently the credit score requirement went up."},
        {"id": "SM-04", "week": 4, "region": "national", "text": "EMIs going up again after the rate hike news, ugh."},
        {"id": "SM-05", "week": 6, "region": "Pune", "text": "Suvidha Finserv follow-up has been slow this month compared to before."},
        {"id": "SM-06", "week": 2, "region": "Nashik", "text": "Quick and easy loan process at my local branch, no complaints."},
        {"id": "SM-07", "week": 7, "region": "Pune", "text": "Still can't believe the flooding a couple weeks back, whole area was shut."},
        {"id": "SM-08", "week": 6, "region": "Pune", "text": "Switched to a competitor because of the zero down payment deal, sorry Suvidha."},
    ]

        # Tier 2: intentionally low-information evidence for the natural ambiguity scenario.
    ambiguous_dealer_notes = [
        {
            "id": "ADN-01",
            "week": 7,
            "branch": "Aurangabad-Cidco",
            "region": "Aurangabad",
            "text": "Dealer reported mixed customer activity and a softer conversion rate this week.",
        },
        {
            "id": "ADN-02",
            "week": 7,
            "branch": "Aurangabad-Waluj",
            "region": "Aurangabad",
            "text": "One experienced coordinator is unavailable; the team is adjusting workloads.",
        },
        {
            "id": "ADN-03",
            "week": 8,
            "branch": "Aurangabad-Garkheda",
            "region": "Aurangabad",
            "text": "Customer traffic has been uneven and several purchase decisions were delayed.",
        },
        {
            "id": "ADN-04",
            "week": 8,
            "branch": "Aurangabad-Cidco",
            "region": "Aurangabad",
            "text": "Some customers are comparing financing options before proceeding.",
        },
    ]

    ambiguous_tickets = [
        {
            "id": "ATK-01",
            "week": 7,
            "region": "Aurangabad",
            "text": "Customer said they are still deciding between financing options.",
        },
        {
            "id": "ATK-02",
            "week": 8,
            "region": "Aurangabad",
            "text": "Customer reported a slower response than expected and asked for a callback.",
        },
    ]

    ambiguous_news = [
        {
            "id": "ANW-01",
            "week": 7,
            "region": "Aurangabad",
            "text": "Local two-wheeler demand remained mixed this month, with dealers reporting uneven showroom activity.",
        },
        {
            "id": "ANW-02",
            "week": 8,
            "region": "Aurangabad",
            "text": "Consumers in the region are comparing financing offers amid a competitive lending environment.",
        },
    ]

    ambiguous_social = [
        {
            "id": "ASM-01",
            "week": 7,
            "region": "Aurangabad",
            "text": "Still deciding which bike loan option makes the most sense.",
        },
        {
            "id": "ASM-02",
            "week": 8,
            "region": "Aurangabad",
            "text": "Loan process felt a little slower this week.",
        },
    ]

    sparse_notes = [
        {
            "id": "SDN-01",
            "week": 2,
            "branch": cfg.SPARSE_BRANCH,
            "region": cfg.SPARSE_REGION,
            "text": "New pilot branch with limited operating history; activity is still being established.",
        }
    ]

    dealer_notes.extend(ambiguous_dealer_notes)
    tickets.extend(ambiguous_tickets)
    news.extend(ambiguous_news)
    social.extend(ambiguous_social)
    dealer_notes.extend(sparse_notes)

    with open(os.path.join(cfg.UNSTRUCTURED_DIR, "dealer_notes.json"), "w") as f:
        json.dump(dealer_notes, f, indent=2)
    with open(os.path.join(cfg.UNSTRUCTURED_DIR, "tickets.json"), "w") as f:
        json.dump(tickets, f, indent=2)
    with open(os.path.join(cfg.UNSTRUCTURED_DIR, "news.json"), "w") as f:
        json.dump(news, f, indent=2)
    with open(os.path.join(cfg.UNSTRUCTURED_DIR, "social.json"), "w") as f:
        json.dump(social, f, indent=2)

    ground_truth = {
        "primary_cause": "competitor_zero_down_payment_scheme",
        "contributing_causes": [
            "monsoon_flooding_footfall",
            "field_agent_attrition",
            "national_cibil_tightening",
            "national_repo_rate_hike",
        ],
        "note": "Evaluation-only. Not to be read by retrieval or reasoning components.",
    }
    with open(cfg.GROUND_TRUTH_PATH, "w") as f:
        json.dump(ground_truth, f, indent=2)

    return len(dealer_notes), len(tickets), len(news), len(social)


if __name__ == "__main__":
    pune_stats, nashik_stats = build_database()
    dn, tk, nw, sm = build_unstructured()

    print("=== CausalBoard Synthetic Data — Suvidha Finserv ===")
    print(f"Pune baseline avg/week:  {pune_stats[0]:,.0f}")
    print(f"Pune anomaly avg/week:   {pune_stats[1]:,.0f}  ({pune_stats[2]:+.1f}%)")
    print(f"Nashik baseline avg/week:{nashik_stats[0]:,.0f}")
    print(f"Nashik anomaly avg/week: {nashik_stats[1]:,.0f}  ({nashik_stats[2]:+.1f}%)")
    print(f"Unstructured records: dealer_notes={dn}, tickets={tk}, news={nw}, social={sm}")
    print("Candidate causes injected: competitor scheme, flooding, agent attrition, CIBIL tightening, repo rate hike")
    print(f"Database written to: {cfg.DB_PATH}")
    print(f"Ground truth (evaluation-only) written to: {cfg.GROUND_TRUTH_PATH}")
