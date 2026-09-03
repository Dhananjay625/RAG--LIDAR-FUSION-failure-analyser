# Changelog

## 2026-09-02 — Fix agent latency: unbounded retry loop, dead model, blocking calls

All changes in `src/agent.py` (only file touched). Backup of the pre-change file kept
at `scratchpad/agent.py.bak`. +25 lines net.

- `src/agent.py:18` — added `import random` (for backoff jitter, below).
- `src/agent.py:46` — model `nvidia/nemotron-3-ultra-550b-a55b` -> `nvidia/nemotron-3-super-120b-a12b`.
  why: ultra failed 3/3 benchmark calls (timeout/503) and was 17-30 s when it did work;
  super averaged 15.4 s with 4/4 valid JSON and confidence 0.85-0.86, which clears the
  0.75 gate so it validates first pass. Both are MoE; active params 55B -> 12B.
- `src/agent.py:165,184` — `"retry_count": retry_count` -> `retry_count + 1` on BOTH
  return paths of `diagnose_node`. **This is the actual hang fix.** LangGraph merges only
  values a node RETURNS, so the counter now persists.
- `src/agent.py:213-226` — `should_retry()`: hard limit `>= 2` -> `>= 3` (preserves
  "initial + 2 retries" now that the counter increments); confidence gate `> 0.75` ->
  `>= 0.75` (an LLM answering exactly 0.75 used to retry); and **deleted
  `state["retry_count"] = new_count`** — the lost-update line that made the guard
  unreachable. Comment added explaining why it must never come back.
- `src/agent.py:144-146` — `self.llm.invoke(prompt)` -> joined `self.llm.stream(prompt)`.
  why: measured time-to-first-token 5.2 s / total 6.3 s streaming vs 15.4 s avg blocking
  on the same prompt.
- `src/agent.py:138-139` — in-node LLM retry ladder `max_retries` 3 -> 2, `wait_times`
  [1,2,4] -> [1,2]. Caps retry amplification at 3x2=6 calls instead of 3x3=9.
- `src/agent.py:174-176` — backoff now `wait_times[attempt] * random.uniform(0.5, 1.5)`
  (full jitter) so concurrent users do not resynchronise on the endpoint.
- `src/agent.py:275-276` — `graph.invoke(initial_state)` -> adds
  `config={"recursion_limit": 10}` as a circuit breaker if the wiring ever regresses.

Verified three ways:
1. Real graph + stubbed LLM pinned at confidence 0.50: terminates after **3** diagnose
   calls (was 5,003 -> GraphRecursionError), k-ladder 5/8/10 intact, HARD STOP fires.
2. Boundary cases on the same harness: conf 0.86 -> 1 call; conf **0.75 -> 1 call**
   (previously retried); conf 0.74 -> 3 calls then hard stop + gate blocks. 
3. Live end-to-end, all 4 built-in scenarios against the real endpoint:
   8.3 s / 7.7 s / 14.4 s / 16.5 s, **mean 11.7 s**, every one single-pass,
   confidence 82-88%, 3 citations each, `safety_passed=True` on all four.

Not changed: `src/retriever.py`, `src/safety.py`, `web/`. Retrieval was never the
bottleneck (22-25 ms embed, 3-5 ms Chroma).


## 2026-09-02 — Corpus expansion to 139 papers + agent latency investigation (no code changed)

**Corpus.** `data/Researchpapers/` 80 -> 139 PDFs (+59, all arXiv, all verified `%PDF`).
Themes added: BEVFusion backbone components (VoxelNet, PointPillars, CenterPoint,
PV-RCNN, VoxelNeXt, BEVDet/4D, BEVDepth, Swin, FPN, Fast-BEV, PETR/v2, M2BEV,
BEVPoolv2); fusion heads (PointPainting, MVP, FUTR3D, AutoAlign/V2, CMT, UVTR,
DeepInteraction, SparseFusion, IS-Fusion, UniTR, BEVFusion4D); SAMFusion backbone
lineage (Gated2Depth, Gated2Gated, Gated Stereo, DETR, Deformable DETR); robustness
+ "how to combine" (MetaBEV, RoboBEV, Robo3D, GraphBEV, RoboFusion, Fusion-is-Not-
Enough, 2 corruption benchmarks); 4 surveys + nuScenes; 10 AEB and 5 CAN-bus papers.
Note SAMFusion main paper was ALREADY present as `2508.16408v1.pdf` (only the supp
was obviously named), and Seeing Through Fog as `1902.08913v3.pdf`.

**Index is now deliberately behind disk:** chroma_db has 14,876 chunks from 80 files;
59 PDFs are un-ingested. This is the SAFE direction (no orphan chunks — verified
`indexed - disk == {}`). Run `venv/bin/python scripts/ingest_new.py` to catch up.

**Model benchmark (2026-09-02, same real RAG prompt, 2,279 chars):**
| model | latency | valid JSON | self-reported confidence |
|---|---|---|---|
| `nemotron-3-ultra-550b-a55b` (in use) | 3/3 calls FAILED (timeout/503) | - | - |
| `nemotron-3-super-120b-a12b` | 6.7/12.7/17.5/24.6 s, avg 15.4 s | 4/4 | 0.85-0.86 |

Super-120b's 0.85 clears the `> 0.75` gate, so it validates first pass and never
enters the retry loop — the win compounds with the counter fix. Streaming on
super-120b: time-to-first-token 5.2 s, total 6.3 s, 55 chunks.
`nemotron-3-nano-30b-a3b` and `qwen3-next-80b-a3b-instruct` return `410 Gone`
(end-of-life) despite still being listed by `available_models` — that list is stale,
always probe before trusting it.

**Agent latency — measured, not guessed. No file under `src/` was modified.**
- Retrieval is NOT the bottleneck: `embed_query` 22-25 ms warm, Chroma k=10 query
  3-5 ms over 14,876 chunks, retriever init 0.4 s.
- The hosted endpoint is: `nvidia/nemotron-3-ultra-550b-a55b` took 15.5 s to return
  the 2 characters "ok"; 5 probes -> 4 OK at 17.3/21.0/25.4/29.7 s, 1x HTTP 503
  "Service temporarily overloaded". So ~22 s floor per LLM call, ~20% 503 rate.
- **BUG (unfixed, reported only): the retry counter never increments.**
  `agent.py:should_retry()` does `state["retry_count"] = new_count` inside a
  conditional-edge function. LangGraph only merges values RETURNED by nodes, so the
  mutation is discarded; `retrieve`/`diagnose` keep seeing `retry_count == 0` and the
  `retry_count >= 2` HARD STOP is unreachable. Reproduced with a stubbed graph mirroring
  agent.py's exact wiring (fake LLM pinned at confidence 0.5): 5,003 diagnose calls
  before `GraphRecursionError` (limit 10007, langgraph 1.2.11). At ~22 s/call that is
  an effectively unbounded hang for ANY query whose self-reported confidence is <= 0.75.
  Repro kept at `scratchpad/graphtest.py`.


## 2026-09-02 — Handle upstream NVIDIA failures instead of reporting them as answers

- `web/server.py` — the agent catches its own exceptions and returns a normal dict
  whose `diagnosis` is an error string, so the API was answering `200 OK` with
  "Error during diagnosis: [503] ..." as the answer. Added `agent_error_message()`
  (regex `AGENT_ERROR_RE`, matched by pattern because the agent owns that wording)
  and `run_with_retries()`: bounded retry (MAX_ATTEMPTS=2) with exponential backoff
  + jitter for transient faults (503/502/504/429/timeout/overloaded), fail-fast on
  permanent ones. Exhausted transient -> HTTP 503; permanent -> HTTP 502, both with
  the upstream text. before: 200 + error-as-answer -> after: correct status + error.
- `web/static/app.js` — string `detail` is no longer JSON.stringify'd (was rendering
  with stray quotes); failed answers keep the question and render a Retry button.
- `web/static/styles.css` — `button.retry`.
- **No files under `src/` were changed.**

Verified: 4 stubbed-agent cases pass (transient->success retries once; exhausted ->
503 after 2 calls; permanent -> 502 with no retry; an answer legitimately starting
with "Error propagation" is NOT flagged). Live run against the real endpoint
returned a cited diagnosis at 88% confidence, HTTP 200.

## 2026-09-02 — Web chat UI for the RAG agent

- `web/server.py` (new) — FastAPI wrapper importing `src.agent.LiDARFailureAnalyzer`.
  `POST /api/ask` runs one question and returns diagnosis + confidence + citations +
  retrieved passages; `GET /api/health` reports agent load state; `/` serves the UI.
  Agent is built lazily once and reused (init is expensive: Chroma + LLM client).
  Sync endpoint on purpose so FastAPI runs the blocking LangGraph call in a threadpool.
- `web/static/{index.html,styles.css,app.js}` (new) — ChatGPT-style chat UI:
  conversation sidebar (localStorage), auto-growing composer, Enter to send,
  per-answer confidence/validation badges, collapsible retrieved-passage panel,
  example prompts, dark/light via `prefers-color-scheme`.
- `run_ui.sh` (new) — starts uvicorn on 127.0.0.1:8000 using ./venv.
- `requirements.txt` — added `fastapi`, `uvicorn[standard]`.
- **No files under `src/` were changed** — the RAG pipeline is untouched.

Verified: server booted on :8123; `/`, `/static/*` → 200; empty question → 422;
real query "Why does LiDAR-camera fusion degrade in fog?" returned a cited
diagnosis at confidence 0.85 against the 19,278-chunk Chroma collection.

## 2026-09-02 — Cull off-topic papers from the corpus and the vector store

- `data/Researchpapers/` — 143 PDFs -> 43 PDFs (100 deleted, ~89MB freed). Removed
  99 papers from a bulk-ingested *Procedia Computer Science* vol. 183 (2021) volume
  (electricity-theft SVMs, fog scheduling, credit scoring, front matter, etc.) plus
  one P&ID symbol-recognition paper. Kept the one on-topic Procedia paper,
  `A-survey-of-LiDAR-and-camera-fusion-enhancemen_2021_*.pdf`. Why: none matched the
  dissertation scope (LiDAR fusion / BEV / SAM) and they dominated retrieval.
  Verified: filename+first-page keyword scan via `pdftotext` over all 143 PDFs;
  final list reviewed by title before deletion.
- `chroma_db` collection `lidar-fusion-papers` — 19,278 chunks -> 9,638 (9,640 stale
  chunks deleted by `filename` metadata). Why: deleting PDFs does NOT evict their
  embeddings; the retriever would have kept serving the deleted papers. Exactly half
  the index was off-topic noise. Verified: index now reports 43 distinct `filename`
  values, matching the 43 PDFs on disk; `LiDARRetriever.retrieve()` on "BEV LiDAR
  camera fusion for 3D object detection" returns 5 on-topic chunks.
- Judgment calls confirmed with the author: kept `978-3-031-73030-6.pdf` (full ECCV
  2024 LNCS 15119 proceedings, 591pp — contains SAMFusion but also ~28 unrelated
  papers) and `1-s2.0-S2214317325000903-main.pdf` (YOLO–SAM for agricultural land use).

## 2026-09-02 — Expand corpus with 37 arXiv papers on BEV + fusion + SAM/foundation models

- `data/Researchpapers/` — 43 -> 80 PDFs (+37, ~270MB), all from arXiv (open access),
  named `arxiv_<id>_<slug>.pdf`. Selected from 164 dedup'd candidates across 12 arXiv
  API queries, scored on BEV / SAM / fusion / LiDAR / AV signals in title+abstract.
  Three groups: canonical BEV view-transform backbones (Lift-Splat-Shoot, BEVFormer
  v1/v2, both BEVFusion papers, BEV survey); BEV multi-sensor fusion (ContrastAlign,
  MapFusion, SimpleBEV, PC-BEV, CoBEVFusion, X-Align/++, DepthFusion, BEVCALIB,
  SB-BEVFusion, weather-occlusion-on-BEVFusion, ...); and the SAM / vision-foundation
  -model-meets-3D bridge (Lift-Splat-Map, SAM3D, SAM-guided pseudo-labels, Multimodal
  SAM-adapter, frozen DINOv2 for BEV, FM-guided BEV maps, MapFM, ViCo3D, FM-OV3D,
  MixSup). Verified: 37/37 downloaded, all `%PDF` magic, no duplicate arXiv IDs
  against the existing corpus.
- `scripts/ingest_new.py` — NEW. Incremental ingest: diffs PDFs on disk against
  `filename` metadata already in Chroma and processes only the difference, so it is
  safe to re-run and never duplicates chunks. Also warns about indexed papers whose
  PDF is gone. Deliberately does NOT modify `src/` (author's RAG code is off-limits);
  it imports `chunker`/`embedder`/`vector_db` as-is. Why: `src/vector_db.py`'s
  `__main__` path re-ingests the whole directory, which would have duplicated all
  9,638 existing chunks.
- `chroma_db` — 9,638 -> 14,876 chunks; 43 -> 80 distinct papers. Verified: index and
  disk agree exactly (0 missing, 0 orphans); retrieval on "BEV feature fusion of LiDAR
  and camera", "sensor-adaptive multimodal fusion in adverse weather" and a SAM+BEV
  query all return on-topic chunks from the new papers.
- Terminology note for the dissertation: "SAMFusion" (Palladin et al., ECCV 2024) means
  Sensor-Adaptive Multimodal Fusion, and "SAMFusion3D" (Glasgow MSc) means Self-Adaptive
  Multi-modality — NEITHER is Meta's Segment Anything Model. The corpus now covers both
  readings of "SAM".

## Current state

Working RAG agent (`src/`) plus a working web UI (`web/`) that surfaces upstream
NVIDIA outages as real HTTP errors with a Retry button. Start with `./run_ui.sh`
and open http://127.0.0.1:8000. Requires Ollama running (`nomic-embed-text`) and
`NVIDIA_API_KEY` in `.env`.

Corpus as of 2026-09-02: `data/Researchpapers/` holds **139** on-topic PDFs; `chroma_db`
(collection `lidar-fusion-papers`) holds 14,876 chunks from only the **first 80** of them.
**59 PDFs are not yet ingested** — run `venv/bin/python scripts/ingest_new.py` to catch up
(it ingests only what is missing). Do NOT run `src/vector_db.py` directly — it re-ingests
the whole directory and duplicates chunks. Chunks carry a `filename` metadata key, so
deleting a PDF requires deleting its chunks too, or the retriever keeps serving a paper
that no longer exists. Un-ingested PDFs are harmless; orphan chunks are not.

FIXED 2026-09-02 (was: unbounded retry loop). `should_retry()` used to increment the
retry counter by mutating `state["retry_count"]` in place; LangGraph merges only values
nodes RETURN, so the write was dropped and the hard stop was unreachable. The counter is
now incremented in `diagnose_node`'s return dicts and `should_retry()` is pure — **do not
reintroduce a mutation there**. A `recursion_limit=10` circuit breaker backs it up.
Model is now `nvidia/nemotron-3-super-120b-a12b` (ultra-550b was failing outright) and
`diagnose_node` streams rather than blocks. Measured: mean 11.7 s over the 4 built-in
scenarios, all single-pass, all safety-passed. Pre-change backup: `scratchpad/agent.py.bak`.

The web wrapper detects agent error strings by regex, so if that wording changes update
`AGENT_ERROR_RE` in `web/server.py`.
