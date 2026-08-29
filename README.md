# Rootwise / CausalBoard

Rootwise is a planned Python prototype for exploring unstructured evidence and identifying possible causal relationships. This repository currently contains **scaffolding only**: no application logic has been implemented.

## Planned implementation approach

The working prototype will ingest demo and user-provided unstructured documents, prepare embeddings for semantic retrieval, and present evidence-backed causal analyses. The backend will expose the analysis workflow through an API, while the frontend will provide an interactive visual interface. Change-point detection and structured validation are planned as supporting analysis components.

## Proposed architecture

```text
data/unstructured/  ->  backend/  ->  vector store and analysis services
                              |
                              v
                         frontend/
```

- `data/unstructured/`: demo documents and other unstructured input data.
- `backend/`: FastAPI services, ingestion, retrieval, and analysis code.
- `frontend/`: Streamlit user interface.
- `tests/`: pytest coverage for the backend and interface-supporting components.
- `deploy/`: deployment configuration for the public prototype.
- `docs/`: architecture notes, design decisions, and user documentation.

## Dependencies

Dependencies are pinned only by package name during the scaffold phase. They cover the planned API, UI, data processing, change-point detection, vector search, embeddings, LLM integration, configuration, and testing stack. See `requirements.txt`.

## Execution instructions

Application entry points are intentionally not present yet. Once implementation begins:

1. Create and activate a Python virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and provide `GEMINI_API_KEY`.
4. Add demo data under `data/unstructured/`.
5. Run the FastAPI backend and Streamlit frontend using the documented entry points.

## Testing

The `tests/` directory is reserved for pytest tests. Test modules will be added alongside implementation.

## Demo data and deployment

Demo input files belong in `data/unstructured/`. Deployment manifests, container definitions, and environment-specific guidance will be added under `deploy/` as the prototype is implemented.

## Repository status

Initial project scaffold for a future public GitHub submission.
