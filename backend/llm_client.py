"""
Provider-agnostic LLM client. Gemini is the initial live provider; an
offline heuristic fallback keeps the demo runnable without network access
or an API key, which matters when recording a demo under time pressure.
"""
import json
import os
from typing import List, Protocol

from dotenv import load_dotenv

from backend.models import EvidenceBundle, Hypothesis

load_dotenv()


class LLMConfigurationError(Exception):
    pass


class HypothesisLLM(Protocol):
    def generate_hypotheses(self, bundle: EvidenceBundle) -> List[Hypothesis]: ...


PROMPT_TEMPLATE = """You are a careful business analyst. Given the evidence below about a \
business KPI anomaly, propose 3 to 5 DISTINCT competing hypotheses that could explain it. \
Do not assume any single cause; consider that multiple factors may coexist. For each \
hypothesis, give: a one-sentence statement, a short mechanism explaining how it would \
cause the observed change, the evidence IDs (from the structured or unstructured evidence \
below) that support it, and any evidence IDs that contradict it. Do NOT provide a \
confidence score or rank them — that is computed separately. Respond ONLY with JSON: \
a list of objects with keys statement, mechanism, supporting_evidence_ids, \
contradictory_evidence_ids, relevant_weeks.

SIGNAL:
{signal}

STRUCTURED EVIDENCE (sample):
{structured}

UNSTRUCTURED EVIDENCE:
{unstructured}
"""


def _build_prompt(bundle: EvidenceBundle) -> str:
    structured_sample = [row.data for row in bundle.structured_evidence[:20]]
    unstructured_list = [
        {"id": d.doc_id, "week": d.week, "region": d.region, "text": d.text}
        for d in bundle.unstructured_evidence
    ]
    return PROMPT_TEMPLATE.format(
        signal=bundle.signal.model_dump_json(indent=2),
        structured=json.dumps(structured_sample, indent=2, default=str),
        unstructured=json.dumps(unstructured_list, indent=2),
    )


class GeminiHypothesisLLM:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key.startswith("your_key"):
            raise LLMConfigurationError("GEMINI_API_KEY is missing or a placeholder. Set it in .env.")
        # Overridable via GEMINI_MODEL in .env in case this model ID is later
        # deprecated (Google has renamed its default Flash model multiple
        # times recently) — avoids needing a code change to recover.
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    def generate_hypotheses(self, bundle: EvidenceBundle) -> List[Hypothesis]:
        from google import genai
        client = genai.Client(api_key=self.api_key)
        prompt = _build_prompt(bundle)
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        raw = json.loads(response.text)
        return [Hypothesis(**h) for h in raw]


class OfflineHeuristicLLM:
    """
    Deterministic, dependency-free fallback used when no API key is present.
    Builds candidate hypotheses directly from which evidence sources are
    non-empty in the bundle, so the pipeline stays demoable offline. This is
    intentionally simple — it is NOT a substitute for the real reasoning
    step, only a safety net for live-demo reliability.
    """

    def generate_hypotheses(self, bundle: EvidenceBundle) -> List[Hypothesis]:
        hypotheses = []
        counts = bundle.source_record_counts
        unstruct = bundle.unstructured_evidence

        if counts.get("competitor_events", 0) > 0:
            hypotheses.append(Hypothesis(
                statement="A competitor's zero-down-payment scheme diverted demand away from Suvidha in the target region.",
                mechanism="Customers deferred or redirected loan applications to a rival offering more favorable terms.",
                supporting_evidence_ids=[d.doc_id for d in unstruct if "competitor" in d.text.lower() or "rival" in d.text.lower() or "zero" in d.text.lower()][:5],
                contradictory_evidence_ids=[],
                relevant_weeks=[bundle.signal.anomaly_start_week],
            ))
        if any(row.source == "footfall" for row in bundle.structured_evidence):
            hypotheses.append(Hypothesis(
                statement="Localized flooding reduced dealership footfall in affected branches.",
                mechanism="Physical access disruption prevented walk-in customers from reaching showrooms.",
                supporting_evidence_ids=[d.doc_id for d in unstruct if "flood" in d.text.lower() or "rain" in d.text.lower() or "waterlog" in d.text.lower()][:5],
                contradictory_evidence_ids=[],
                relevant_weeks=[bundle.signal.anomaly_start_week],
            ))
        if any(row.source == "agents" for row in bundle.structured_evidence):
            hypotheses.append(Hypothesis(
                statement="Field agent attrition reduced on-ground loan sourcing capacity in specific branches.",
                mechanism="Fewer active sourcing agents means fewer loan applications generated and processed.",
                supporting_evidence_ids=[d.doc_id for d in unstruct if "agent" in d.text.lower() or "executive" in d.text.lower() or "staff" in d.text.lower()][:5],
                contradictory_evidence_ids=[],
                relevant_weeks=[bundle.signal.anomaly_start_week],
            ))
        if any(row.source == "cibil_policy" for row in bundle.structured_evidence):
            hypotheses.append(Hypothesis(
                statement="A national CIBIL cutoff tightening reduced approval rates, surfacing first or more strongly in this region.",
                mechanism="Raising the credit-score threshold disqualifies previously eligible applicants.",
                supporting_evidence_ids=[d.doc_id for d in unstruct if "cibil" in d.text.lower() or "credit score" in d.text.lower() or "rejected" in d.text.lower()][:5],
                contradictory_evidence_ids=[],
                relevant_weeks=[bundle.signal.anomaly_start_week],
            ))
        if any(row.source == "repo_rate" for row in bundle.structured_evidence):
            hypotheses.append(Hypothesis(
                statement="The national repo-rate hike raised EMIs and suppressed demand.",
                mechanism="Higher interest rates increase monthly repayment burden, discouraging new loan uptake.",
                supporting_evidence_ids=[d.doc_id for d in unstruct if "rate" in d.text.lower() or "emi" in d.text.lower() or "repo" in d.text.lower()][:5],
                contradictory_evidence_ids=[],
                relevant_weeks=[bundle.signal.anomaly_start_week],
            ))
        return hypotheses[:5] if hypotheses else [Hypothesis(
            statement="Insufficient distinguishing evidence to propose a specific hypothesis.",
            mechanism="No structured or unstructured evidence source showed a clear pattern.",
            supporting_evidence_ids=[], contradictory_evidence_ids=[], relevant_weeks=[],
        )]


def get_llm_client() -> HypothesisLLM:
    """Returns the live Gemini client if configured, else the offline fallback."""
    try:
        return GeminiHypothesisLLM()
    except LLMConfigurationError:
        return OfflineHeuristicLLM()
