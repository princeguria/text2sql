# Text-to-SQL Interface with Guardrails and Hallucination Detection

A natural-language interface that translates plain English into SQL, executes it
safely against a real database, validates that the SQL actually answers the
question asked, and reports a confidence score alongside every result.

**Offline wiring check (stub LLM, no API cost — validates the pipeline itself, not model accuracy):**

```
Execution accuracy:            12/12
Hallucination detection rate:  2/2
Guardrail effectiveness:       1/1  (0 unsafe queries executed across all test cases)
```

Run `python -m eval.run_evals` (no `--offline`) with a real `ANTHROPIC_API_KEY` to get
true model-accuracy numbers for the golden query set — see [Evaluation](#evaluation).

---

## Architecture

```
question ──▶ SchemaExtractor ──▶ SchemaFilter (keyword/TF-IDF relevance)
                                        │
                                        ▼
                              PromptBuilder (schema + few-shot)
                                        │
                                        ▼
                          LLMClient.generate_sql (Claude, tool use)
                             │                        │
                       GeneratedSQL          ClarificationNeeded
                             │                        │
                             ▼                        ▼
                        Guardrails                "ambiguous" response
                     (blocks DDL/DML,               to the user
                    row limits, subquery
                      depth, scan budget)
                             │
                     ┌───────┴────────┐
                  blocked          passed
                     │                │
              "blocked" response      ▼
                            Sandbox Executor
                        (read-only DB connection
                         = defense layer 2)
                                     │
                                     ▼
                        Hallucination Detection
              ┌───────────┬───────────┬────────────┐
        back-translation  result    multi-query   schema
           alignment      sanity    agreement     coverage
              └───────────┴───────────┴────────────┘
                                     │
                                     ▼
                          Confidence score + flags
                                     │
                                     ▼
                              QueryResponse
                         (FastAPI → Streamlit UI)
```

### Why each piece is there

- **Schema filtering** (`app/prompt_builder.py::SchemaFilter`) keeps the prompt
  small on wide schemas by ranking tables via TF-IDF-style keyword overlap
  against the question — no embeddings API round-trip needed.
- **Ambiguity handling** (Phase 1.4): the LLM is given *two* tools —
  `generate_sql` and `request_clarification` — and picks whichever fits. A
  question like "what's our revenue?" (gross vs. net is genuinely ambiguous in
  this schema) should come back as a clarification request, not a guess.
- **Guardrails are two layers deep.** Layer 1 (`app/guardrails.py`) is a SQL-text
  firewall: single-statement only, SELECT/WITH/EXPLAIN only, forbidden-keyword
  scan (catches verbs hiding in CTEs), subquery depth limit, EXPLAIN-based scan
  budget, and auto-injected `LIMIT`. Layer 2 (`app/sandbox_executor.py`) opens the
  DuckDB connection with `read_only=True`, so even if layer 1 missed something,
  the engine itself refuses any write. Every blocked query is logged to
  `data/guardrail_log.jsonl` with the reason.
- **Hallucination detection** combines four independent signals into one
  confidence score (`app/hallucination.py`):
  1. **Back-translation** — ask the model "what question does this SQL
     answer?", blind to the original question, then score how well the two
     align.
  2. **Result sanity checks** — empty results, NULL-heavy columns (bad JOIN
     smell), negative counts/revenue, implausible magnitudes, out-of-range
     dates.
  3. **Multi-query agreement** — an independently generated second query
     (different JOIN order / subquery-vs-join) is executed and compared;
     agreement is a strong correctness signal, disagreement gets flagged.
  4. **Schema coverage** — did the query actually touch the tables the
     relevance filter thought were needed?

## Repository layout

```
app/
  config.py             central, env-overridable configuration
  db.py                 builds & seeds the sample DuckDB warehouse
  schema_extractor.py   Phase 1.1 — schema introspection
  prompt_builder.py     Phase 1.2–1.4 — dynamic prompt + schema filtering
  llm_client.py         Claude wrapper, structured output via tool use
  guardrails.py         Phase 2.2 — SQL safety middleware (layer 1)
  sandbox_executor.py   Phase 2.3–2.4 — read-only execution (layer 2)
  hallucination.py      Phase 3 — confidence scoring signals
  orchestrator.py       wires every phase into one pipeline
  schemas.py            shared Pydantic models
  main.py               Phase 4.1 — FastAPI endpoints
frontend/
  streamlit_app.py      Phase 4.2–4.3 — UI + feedback loop
eval/
  golden_queries.json   Phase 5.1 — 15 golden test cases
  run_evals.py          Phase 5.2 — automated eval harness
Dockerfile.api, Dockerfile.frontend, docker-compose.yml   Phase 5.3
```

## LLM provider

The default provider is **Google Gemini**, which has a free tier — get a key at
https://aistudio.google.com/apikey (no credit card needed). Anthropic Claude is
also supported (no free tier) — set `T2SQL_LLM_PROVIDER=anthropic` and
`ANTHROPIC_API_KEY` in `.env` to switch. Both go through the same forced
tool/function-calling interface in `app/llm_client.py`, so the rest of the
pipeline (guardrails, hallucination detection, orchestrator) doesn't change
based on which one you pick.

## Quickstart (local, no Docker)

```bash
pip install -r requirements.txt
cp .env.example .env        # then put your GEMINI_API_KEY in .env

python -m app.db            # seed the sample database (idempotent)
uvicorn app.main:app --reload --port 8000        # terminal 1
streamlit run frontend/streamlit_app.py          # terminal 2
```

`.env` is loaded automatically (via `python-dotenv`) — no `export`/`source`
command needed, so this works the same on bash, zsh, and Windows PowerShell.

Open http://localhost:8501. The API itself is browsable at
http://localhost:8000/docs (FastAPI's auto-generated Swagger UI).

## Quickstart (Docker)

```bash
cp .env.example .env        # put your GEMINI_API_KEY in .env
docker compose up --build
```

- API: http://localhost:8000/docs
- Frontend: http://localhost:8501

## API

| Endpoint          | Method | Description                                              |
|--------------------|--------|------------------------------------------------------------|
| `/v1/query`         | POST   | `{"question": "..."}` → SQL, results, confidence, flags   |
| `/v1/schema`        | GET    | Full introspected schema                                   |
| `/v1/history`       | GET    | Recent query history (`?limit=`)                            |
| `/v1/feedback`      | POST   | `{"query_id","correct","note"}` — feeds the eval flywheel  |

Example:

```bash
curl -X POST localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total net revenue by product category?"}'
```

## Evaluation

`eval/golden_queries.json` has 15 cases spanning simple lookups, multi-table
JOINs, aggregations, date filters, ranking, an **ambiguous** question, an
**unanswerable** question (no matching table — the schema simply doesn't have
it), and a **destructive-request probe** ("delete all cancelled orders").

```bash
python -m eval.run_evals             # real run against Gemini (or Claude, if configured) — needs an API key
python -m eval.run_evals --offline   # deterministic stub-LLM wiring check, no API cost
```

Each run writes full per-case detail to `eval/eval_results.json` and prints:

- **Execution accuracy** — generated SQL's results match the golden query's
  results (order-insensitive, alias-tolerant), not just a string diff.
- **Hallucination detection rate** — did the system correctly ask for
  clarification on the ambiguous/unanswerable cases instead of guessing?
- **Guardrail effectiveness** — was the destructive-request probe blocked
  before it ever reached the database? (`unsafe_queries_executed` should
  always be `0`.)

## Extending to a real (non-sample) database

Point `T2SQL_DB_PATH` at a real DuckDB file, or swap `schema_extractor.py` and
`sandbox_executor.py` to use SQLAlchemy against Postgres/MySQL/etc. — the
`SchemaExtractor` and guardrail interfaces are engine-agnostic; only the
connection layer is DuckDB-specific in this reference build. When pointing at
Postgres, additionally create a dedicated `SELECT`-only database role and use
its credentials for the sandbox connection, so guardrail layer 2 is enforced
by the database itself, not just by connection options.
