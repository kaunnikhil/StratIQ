"""
Evidence retrieval layer.

NOTE ON DESIGN TRADEOFF: this uses TF-IDF + cosine similarity (scikit-learn)
rather than dense embeddings (sentence-transformers/Chroma). This was a
deliberate call under a hard deadline: the dense-embedding stack hit a
Windows OpenMP/PyTorch DLL conflict that cost significant debugging time
with no guaranteed fix. TF-IDF has zero native-dependency risk, installs in
seconds, and is fully sufficient for keyword/entity-rich business text like
dealer notes, tickets, and news snippets. Swapping in dense embeddings later
is a drop-in replacement behind the same `search()` interface.
"""
import glob
import json
import os
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.models import UnstructuredDoc


class EvidenceVectorStore:
    def __init__(self, persist_dir: str = "data/index"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self._docs: List[dict] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None

    def index_json_sources(self, unstructured_dir: str, time_origin: Optional[str] = None) -> int:
        """Loads every *.json file in unstructured_dir and builds the TF-IDF index."""
        self._docs = []
        for path in sorted(glob.glob(os.path.join(unstructured_dir, "*.json"))):
            source_name = os.path.splitext(os.path.basename(path))[0]
            with open(path) as f:
                records = json.load(f)
            for rec in records:
                doc = {
                    "doc_id": rec.get("id", f"{source_name}-{len(self._docs)}"),
                    "source": source_name,
                    "week": rec.get("week"),
                    "region": rec.get("region"),
                    "branch": rec.get("branch"),
                    "text": rec.get("text", ""),
                }
                self._docs.append(doc)

        if not self._docs:
            raise ValueError(f"No documents found in {unstructured_dir}")

        corpus = [d["text"] for d in self._docs]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(corpus)
        return len(self._docs)

    def search(self, query: str, top_k: int = 10,
               week_min: Optional[int] = None, week_max: Optional[int] = None,
               region: Optional[str] = None) -> List[UnstructuredDoc]:
        if self._vectorizer is None or self._matrix is None:
            raise RuntimeError("Index not built. Call index_json_sources() first.")

        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix).flatten()

        scored = []
        for i, score in enumerate(sims):
            doc = self._docs[i]
            if week_min is not None and doc["week"] is not None and doc["week"] < week_min:
                continue
            if week_max is not None and doc["week"] is not None and doc["week"] > week_max:
                continue
            if region is not None and doc["region"] is not None and doc["region"] != region and doc["region"] != "national":
                continue
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored[:top_k]:
            results.append(UnstructuredDoc(
                doc_id=doc["doc_id"], source=doc["source"], week=doc["week"],
                region=doc["region"], branch=doc["branch"], text=doc["text"],
                relevance=round(float(score), 4),
            ))
        return results
