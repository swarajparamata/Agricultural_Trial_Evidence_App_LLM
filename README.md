# AgriEvidence LLM service

The Evidence Assistant of AgriEvidence. It is a LangGraph agent that answers questions about agricultural trial data with **Qwen 3.6 running in Ollama**, and it also extracts fields from free-text trial reports.

The service is stateless. Its only client is the backend (`Agricultural_Trial_Evidence_App_Backend`), which sends the reconciled trials with every request, so this service has no database of its own.

> **Part of AgriEvidence.** To run the whole app with one command on macOS or Windows, clone [Agricultural_Trial_Evidence_App](https://github.com/swarajparamata/Agricultural_Trial_Evidence_App) with `git clone --recursive`. Its README also covers the model configuration and the demo data.

**Stack:** Python 3.10+ · FastAPI · LangChain (`langchain-core`, `langchain-ollama`) · LangGraph · Ollama

## Getting started

Requires Python 3.10 or newer.

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # optional: every setting has a working default
python -m agritrial_llm           # http://127.0.0.1:8100, interactive docs at /docs
```

```powershell
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m agritrial_llm
```

Without Ollama the service still answers, using the **offline planner** (see [below](#the-offline-planner)). This keeps demos working on any laptop.

## Ollama and Qwen 3.6

1. Install Ollama from <https://ollama.com/download>. On macOS you can also run `brew install ollama` and then `ollama serve`. It listens on `http://127.0.0.1:11434`, and on Apple Silicon it runs models on the GPU.
2. Pull the model and match it in `.env`:

   | Tag | Size | `.env` |
   | --- | --- | --- |
   | `qwen3.6` (35B-A3B, mixture of experts) | ~24 GB | `OLLAMA_MODEL=qwen3.6` (default) |
   | `qwen3.6:27b` | ~17 GB | `OLLAMA_MODEL=qwen3.6:27b` |

   ```powershell
   ollama pull qwen3.6
   ```

3. Check with `curl http://127.0.0.1:8100/health`. You should see `"reachable": true` and `"model_available": true`.

Qwen 3.6 needs roughly 17–24 GB of RAM or VRAM. Current tags and sizes are listed at <https://ollama.com/library/qwen3.6/tags>. On a smaller machine you have three options:
- point `OLLAMA_BASE_URL` at a GPU server that runs Ollama;
- use a smaller Ollama model that supports tool calling, for development;
- rely on the offline planner.

Admins can change the model and the Ollama URL at runtime under **Settings → AI assistant**. The backend sends them with each request, and the same page has a **Test the model** button.

Qwen 3.6 is a thinking model. `LLM_THINK=true` improves hard questions but is much slower on a CPU.

### Small models

A small tool-calling model such as `qwen3:1.7b` (1.4 GB) runs on an ordinary laptop CPU, at about 1–4 minutes per question. To use one:
- `ollama pull qwen3:1.7b`;
- set `OLLAMA_MODEL=qwen3:1.7b` (and `OLLAMA_NUM_CTX=8192` to save memory) in `.env`, or set the model in Settings → AI assistant.

Small models copy example values, guess IDs and invent caveats. The agent guards against this:
- **Parser hint:** the prompt carries the rule-based parser's reading of the question as suggested first calls, with names already resolved.
- **Tool guards:** calls are refused, with a corrective message, when they:
  - use trial IDs the model never saw;
  - use filters the question doesn't mention;
  - put a name in the wrong field (`country='Rice'`).
- **Tool results:** they always list a trial's caveats, even when empty. When a combination finds nothing, they say what each part finds on its own.
- **Fact-check:** it also rejects caveats given to the wrong trial and blanket claims that don't fit the cited trials.

Answers from a 1.7B model are still less reliable than from Qwen 3.6. For a presentation without a big machine, "Always use the offline planner" gives predictable answers.

## How the agent works: two loops

![The agent graph: agent and tools form the tool loop; verify, revise and agent form the reflection loop; verify leads to finalize and the answer](docs/agent-loops.png)

The graph is built in `agritrial_llm/agent.py`. The image is rendered from `docs/agent-loops.mmd`, and `GET /v1/agent/graph` returns the live graph as Mermaid, generated from the compiled graph.

**Tool loop (`agent ⇄ tools`):**
- The model, with the tools bound, decides which tools to call. The results go back to it as tool messages, and it repeats until it has enough evidence.
- The loop is bounded by `AGENT_MAX_TOOL_ITERATIONS` model calls per round. When the budget is used up, the model is asked for its final answer without tools.
- Tool errors, such as bad arguments, are returned to the model so it can correct itself.

**Reflection loop (`verify → revise → agent`):**
- The draft is checked deterministically against the tool results:
  - every number must appear in the tool results (rounding and unit conversions are allowed);
  - every trial ID must exist;
  - conflicts and missing controls in the cited trials must be disclosed;
  - data answers must be based on tool results.
- Failed checks go back to the model as reviewer feedback, up to `AGENT_MAX_REVISIONS` times. If problems remain after the last revision, the answer is returned with a visible fact-check note.

Every step is streamed as an event. The app shows these events in the chat's **Inspection** tab.

### Tools

| Tool | What it returns |
| --- | --- |
| `search_trials` | Trials matching crop, product, country, year, trial type, data status or free text. Each field takes one value or a list (`country=['France', 'Germany']`), and years also take a range (`year_from`, `year_to`). Spelling variants such as `HARVESTPLUS`, `potatoes` or `FR` are resolved. |
| `get_trial_details` | One trial with every source file and the values it reported, conflicts, notes and caveats. |
| `trial_statistics` | Mean, median, min and max uplift. Also: trials excluded for a missing yield, the range caused by conflicting sources, and an optional `group_by` ranking. Ties list every trial. |
| `compare_trials` | Fields that differ, a ranking by uplift (tied trials share a place), and warnings about different crops, conflicts or missing controls. |
| `find_data_issues` | Conflicts, missing values, single-source trials and caveat counts. |
| `list_reference_data` | Products, crops, countries, years and trial types with their spelling variants and counts, plus the custom fields. |

### The offline planner

`RuleBasedChatModel` (`offline.py`) is a LangChain chat model. It plans tool calls from the question with rules (`query.py`) and writes the answer from templates (`composer.py`).

Like the model, it chooses its next step from the tool results. After the first round it can run a second:
- when a combination finds nothing, such as Harvest Plus on rice, it searches for each part on its own, so the answer says what the data does contain;
- for an evidence question, it reads every conflicting trial in full, so the answer can quote the sources.

It runs inside the **same graph**, with the same tools and checks, so the inspection log looks the same as with Qwen. The service uses it when:
- Ollama is unreachable or the model isn't installed;
- an LLM call fails mid-run;
- `force_offline` is set (Settings → AI assistant → "Always use the offline planner", useful for presentations without a GPU).

`AGENT_OFFLINE_FALLBACK=false` turns the fallback off. A missing model then returns HTTP 503.

### Report extraction loop (`extract → validate → extract`)

When an imported report doesn't match the rule-based parser, the backend calls `POST /v1/extract/report`. Qwen fills a structured schema (`with_structured_output`, JSON schema mode), and `validate_extraction` checks it:
- the required fields are present;
- the trial ID appears in the text or the file name;
- both yields appear in the text exactly as written;
- the year is plausible;
- the unit is known.

Problems go back to the model for another attempt, up to `EXTRACTION_MAX_ATTEMPTS`. This path needs the LLM; the offline planner doesn't extract reports.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service status, whether Ollama is reachable and has the model (`?model=` and `?ollama_base_url=` override the defaults), and the default options. |
| `POST` | `/v1/agent/chat` | Question + history + dataset + options → answer, mode (`llm`/`offline`), cited trials and sources, loop counters, verification and the full trace. |
| `POST` | `/v1/agent/stream` | The same as server-sent events: `log` events while the agent works, then `answer` (or `error`) and `done`. |
| `POST` | `/v1/extract/report` | Free-text report → validated fields. |
| `POST` | `/v1/llm/test` | Sends a one-line prompt to the model and reports the reply and the latency. |
| `GET` | `/v1/agent/graph` | Mermaid diagrams of the agent graph and the extraction graph. |

The request and response schemas are in `agritrial_llm/schemas.py` and at `/docs`. Per-request `options` override the defaults below: model, Ollama URL, temperature, context size, thinking, loop limits, reflection, fallback, `force_offline`, history turns and extra instructions.

If `LLM_SERVICE_API_KEY` is set, every `/v1` request must send it in the `X-API-Key` header. Set the same value as `LLM_SERVICE_API_KEY` in the backend's `.env`.

Example with the bundled test dataset (run from this folder):

```python
import json, httpx

dataset = json.load(open("tests/fixtures/sample_dataset.json", encoding="utf-8"))
reply = httpx.post(
    "http://127.0.0.1:8100/v1/agent/chat",
    json={"question": "Which product performs best?", "dataset": dataset, "options": {"force_offline": True}},
    timeout=300,
).json()
print(reply["mode"], reply["verification"]["passed"])
print(reply["answer"])  # **Harvest Plus** has the highest mean yield uplift: **+6.3%**, ahead of Root Boost at +2.5%. …
```

## Configuration

Settings come from environment variables or `.env` in this folder (see `.env.example`).

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_SERVICE_HOST` / `LLM_SERVICE_PORT` | `127.0.0.1` / `8100` | Where `python -m agritrial_llm` listens. The backend's `LLM_SERVICE_URL` must point here. |
| `LLM_SERVICE_API_KEY` | empty | Shared secret for `/v1` requests (`X-API-Key`). |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server. Use `127.0.0.1` rather than `localhost` on Windows: `localhost` tries IPv6 first and doubles the wait when Ollama is down. |
| `OLLAMA_MODEL` | `qwen3.6` | Model tag. |
| `OLLAMA_TIMEOUT_S` | `180` | Timeout of one model call. |
| `OLLAMA_NUM_CTX` | `16384` | Context window. |
| `OLLAMA_KEEP_ALIVE` | `10m` | How long Ollama keeps the model loaded. |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature. |
| `LLM_THINK` | `false` | Qwen thinking mode. |
| `AGENT_MAX_TOOL_ITERATIONS` | `6` | Model calls per round before the agent must answer. |
| `AGENT_MAX_REVISIONS` | `2` | Reflection rounds after failed checks. |
| `AGENT_ENABLE_REFLECTION` | `true` | Run the verify/revise loop. |
| `AGENT_OFFLINE_FALLBACK` | `true` | Use the offline planner when the model is unavailable. |
| `EXTRACTION_MAX_ATTEMPTS` | `3` | Attempts of the report extraction loop. |

## Tests

```powershell
python -m pytest
```

The 86 tests need neither Ollama nor the backend. The tests point Ollama at a closed port and drive the LLM path with a scripted chat model that implements the LangChain interface. They cover:
- the tool loop and its budget;
- reflection after an ungrounded answer;
- stopping after the last revision;
- fallback when the model fails mid-run;
- the offline answers on the sample data;
- the grounding checks;
- the extraction loop;
- the HTTP API, including SSE and the API key;
- the shape of the backend's payload (`tests/fixtures/backend_dataset.json`).

The tests do not run a real Qwen model. Check that path on a machine that has Ollama, using `/health` and `/v1/llm/test`.

## Project layout

```
agritrial_llm/
├── main.py          # FastAPI app and endpoints
├── __main__.py      # python -m agritrial_llm (uvicorn)
├── config.py        # ServiceSettings (.env)
├── schemas.py       # request/response models, dataset contract with the backend
├── agent.py         # LangGraph agent: tool loop + reflection loop, run/stream helpers
├── extraction.py    # LangGraph extraction loop for free-text reports
├── tools.py         # the six LangChain tools
├── store.py         # EvidenceStore: vocabulary resolution, filters, statistics, comparison
├── verification.py  # deterministic grounding checks that drive the reflection loop
├── llm.py           # ChatOllama setup, Ollama health probe, option resolution
├── offline.py       # RuleBasedChatModel (offline planner)
├── query.py         # question analysis for the offline planner
├── composer.py      # answer templates for the offline planner
├── prompts.py       # system, revision and extraction prompts
└── events.py        # step events for the inspection log
tests/               # pytest suite and fixtures
```
