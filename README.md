# Sekiro Lore & Wiki RAG Assistant

A retrieval-augmented chatbot that answers questions about *Sekiro: Shadows Die Twice* using
only an indexed corpus of the game's wiki. Every answer cites the pages it came from.

![The assistant answering an obscure lore question, with the cited wiki sections expanded](docs/screenshots/03-cited-sources.png)

Asked without retrieval, the same question invents an "Ashina Blade" and a betrayal that never
happened. With retrieval, the citations point at the real page. See [RAG evaluation](#rag-evaluation).

| Track | What it does | Status |
|---|---|---|
| **Core** | Wiki RAG pipeline — ingest, chunk, embed, retrieve, generate, cite | ✅ complete |
| **Extended** | Boss detection from screenshots across all 40 boss classes, biases retrieval toward the detected boss | ✅ fine-tuned, weights ship with the repo |

---

## Architecture

The backend loads every heavyweight dependency **once at startup** (FastAPI `lifespan`) and
reuses it per request — nothing is rebuilt on the fly.

```
                 ┌────────────────────────┐
                 │   Streamlit frontend    │  reads API_BASE_URL from env
                 └───────────┬────────────┘
                             │ HTTP
                 ┌───────────▼────────────┐
                 │    FastAPI backend      │
                 └──┬──────────┬────────┬──┘
                    │          │        │
      ┌─────────────▼──┐  ┌────▼─────┐ ┌▼────────────────┐
      │RetrievalService │  │Generation│ │DetectionService  │
      │ MiniLM + Chroma │  │  Ollama  │ │ YOLO (optional)  │
      └────────┬────────┘  └──────────┘ └──────────────────┘
      ┌────────▼─────────┐
      │ 627 chunks from  │
      │ 172 wiki pages   │
      └──────────────────┘
```

Offline indexing (run once, in the notebook): raw wikitext → clean → section-chunk (~400
tokens, ~75 overlap) → tag boss chunks with their YOLO class → embed (`all-MiniLM-L6-v2`) →
persist to ChromaDB.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` | 384-dim, fast on CPU, strong on short passages |
| Vector store | ChromaDB (persistent, cosine) | Zero-infra local persistence + metadata filtering |
| LLM | Ollama, `qwen2.5-7B-instruct` Q4_K_M | Fully local, no API key, swappable via env var |
| Backend | FastAPI + Uvicorn | Async, typed schemas, lifespan hooks |
| Frontend | Streamlit | Single-file chat UI, no build step |
| Detection | Ultralytics YOLOv8n (optional) | Extended Track — fine-tuned on 40 boss classes |

---

## Repository structure

```
sekiro-wiki-assistant/
├── notebooks/               rag_pipeline.ipynb (graded), yolo_boss_detection.ipynb,
│                             wiki_preprocess.py, vector_store_export/
├── backend/app/              main.py, api/routes/query.py, services/{retrieval,generation,detection}.py
│   ├── tests/test_query.py   12 tests
│   ├── data/vector_store/    committed
│   └── data/yolo_model/best.pt   committed, ~6 MB
├── frontend/app.py           chat UI + api_client.py + assets/
├── docs/                     capture_screenshots.py + screenshots/
├── DataPreparing Tools/      extract_frames.py, filter_boss_frames.py, train_boss40.py
├── data/raw/raw_wiki/        219 source .txt files (not committed)
├── data/images/              raw footage + dataset exports (not committed)
└── runs/                     [Extended] training output (not committed)
```

---

## Domain and data

219 wikitext pages scraped from the [Sekiro Fandom wiki](https://sekiro-shadows-die-twice.fandom.com/wiki/Sekiro:_Shadows_Die_Twice_Wiki)
as raw MediaWiki source. 46 were redirects and skipped, leaving 172 indexed pages and 627
chunks (400/75 tokens, min 40). Covers bosses, endings, items, prosthetics, combat arts, NPCs,
locations and lore. Full per-page breakdown: [`CORPUS.md`](CORPUS.md).

Boss-page chunks are tagged with the matching YOLO class name so retrieval can be filtered to
the boss on screen. The detector itself names **40 classes**; these 5 are the ones that have a
wiki page to filter against, which is why the table below has 5 rows:

| Class | Tagged chunks |
|---|---|
| `corrupted_monk` | 20 |
| `divine_dragon` | 5 |
| `genichiro` | 18 |
| `guardian_ape` | 12 |
| `owl` | 26 |

---

## Extended Track: dataset preparation

The detector is trained on **all 40 bosses of the base game** — not a hand-picked subset.
Building that set went through six stages, in this order.

**1. Footage.** Three sources, mixed on purpose so the model does not learn one player's
habits or one arena's camera: our own gameplay recordings, YouTube walkthroughs, and
speedrunner runs. The speedrunner footage is the least normal of the three and earns its place
for exactly that reason — those fights are closed out fast, from odd angles, with the boss at
distances an ordinary playthrough never produces.

**2. Frame extraction.** Frames were pulled from every video programmatically
(`extract_frames.py`) rather than grabbed by hand. 40 classes cannot be covered by scrubbing
timelines, and hand-grabbing biases a set toward whatever happens to look good paused.

**3. Cleaning.** Menus, loading screens, death screens, burn-in subtitles and blurry or
near-duplicate frames were dropped (`filter_boss_frames.py`), leaving frames with a real boss
on screen and a readable silhouette.

**4. Boss selection.** Every boss in the base game became a class — 40 of them, from `Armored
Warrior` to `Tokujiro the Glutton`. `Genichiro` is the single boss split across two classes
(`Genichiro Phase 1`, `Genichiro Phase 2`); the other 39 are one class per boss.

**5. Balancing — a 40-photo target per boss.** Raw footage is hopelessly skewed: a boss that
gates progress gets fought for an hour while a miniboss gets fought twice, so an unbalanced
extract is hundreds of images of one fight and a handful of another. The set was therefore
balanced so that every boss holds on average **40 photos in total across train, valid and
test**. That target is what makes 40 classes trainable together instead of the head classes
crowding out the tail. 1,352 images resulted:

| split | images |
|---|---|
| train | 947 |
| valid | 271 |
| test | 134 |

**6. Labelling on Roboflow.** The balanced set was uploaded to **Roboflow** and labelled
there — boxes drawn in the browser against the 40-name class list — then exported in YOLOv8
format (project `Sekiro Boss Detection Cap 40`, v2, 15 Sep 2026, CC BY 4.0). The export
applies no augmentation, only EXIF auto-orientation. Geometry stays native 16:9 and
Ultralytics letterboxes to 640 at both train and predict time, so nothing is stretched.

Because the class list runs the whole game, it includes bosses that appear in a single arena
for a single fight, and the rarest of them are the thinnest part of the set. The corpus side
reflects that honestly: only the 5 bosses with a wiki page can filter retrieval, and the other
35 are named in the response without narrowing the search.

---

## Training & evaluation (Extended Track)

Fine-tuned from pretrained `yolov8n.pt` on the 40-class export: 150 epochs maximum, patience
40, imgsz 640, batch 8, seed 0, AMP, on an RTX 3050 Ti (4GB). Classification loss was
up-weighted (`cls=1.0`, `box=7.5`, `dfl=1.5`) because across 40 imbalanced classes the failure
to avoid is a confident wrong class, not a loose box. Early stopping fired at epoch **125** —
exactly the best epoch (85) plus the patience window.

Best checkpoint, which is the `best.pt` that ships:

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| valid (early-stop signal, 271 images) | 0.858 | 0.680 | 0.826 | 0.476 |

`runs/boss40_v1/weights/best.pt` and `backend/data/yolo_model/best.pt` are byte-identical
(SHA-256 `915ea66e…`), so the table above describes the weights the backend actually loads.

**mAP50-95 sits far below mAP50** (0.48 against 0.83). Boxes land in the right place but are
not tightly drawn, and for this application that is the cheapest possible weakness: the
Extended Track only needs the predicted class name to become a metadata filter, never the box
coordinates.

**Reading the precision/recall split.** The model is more precise than it is exhaustive — it
misses bosses more often than it misnames them, which is the right way round here. The shipped
`YOLO_CONFIDENCE_THRESHOLD` is **0.50** for the same reason: a missed detection falls back to
unfiltered retrieval and still answers, while a wrong detection would confidently filter the
answer to the wrong boss.

Only 5 of the 40 classes can filter retrieval — the 5 with tagged wiki pages (table above).
The other 35 are returned as stable slugs so the response still names the boss on screen, and
`retrieve()` falls through to unfiltered search rather than returning nothing.

A silent trap worth knowing: Ultralytics reads a raw numpy array as BGR and a PIL image as RGB.
Passing a numpy array of an RGB image silently swaps red/blue and can flip the predicted class
with no error — confirmed on a held-out dragon frame (PIL → `divine_dragon` 0.68; numpy → `owl`
0.35). `DetectionService.detect()` passes the PIL image directly.

The run's own output — curves, confusion matrix, `results.csv` — is in `runs/boss40_v1/`
(gitignored, present after training). The notebook's Extended Track sections document the
earlier 5-class iteration; the BGR investigation and the integration notes still apply.

---

## RAG evaluation

Ten spec questions (three deliberately obscure) were run end to end.

| Kind | Result |
|---|---|
| Well-known (7) | 5 correct, 2 refused despite the answer being in context (generation-side misses, not retrieval failures) |
| Obscure (3) | 3/3 correct |
| Corpus gap (1 of the well-known 7) | Correctly refused — no page on "Dragon's Heritage" exists in the scrape |

**The grounding proof:** the same three obscure questions, asked with no retrieved context,
make up a different game's lore — Kuro becomes a "dog companion," the Sculptor gets an invented
"Ashina Blade" and clan betrayal, Emma becomes "Genichiro's wife." None of it is in the corpus,
all of it stated with full confidence, and it isn't even stable across runs. That instability is
the strongest argument for grounding: there's no memorized answer to fall back on, so without
the corpus the model just invents one.

Full question-by-question transcripts and the two generation-side failure cases are in
`notebooks/rag_pipeline.ipynb`, section 5.6.

---

## Setup

```bash
# Prerequisites: Python 3.12+, Ollama running
ollama pull qcwind/qwen2.5-7B-instruct-Q4_K_M

# Backend
cd backend
python -m venv .venv && . .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py   # open http://localhost:8501

# Tests
cd backend && pytest -v
```

The vector store is committed at `backend/data/vector_store/`; regenerate it by running
`notebooks/rag_pipeline.ipynb` and copying `notebooks/vector_store_export/` over it.

Boss detection is **optional and non-fatal** — no weights, no `ultralytics`, no problem;
`/health` reports it disabled and the Core Track keeps serving. Comment out `YOLO_MODEL_PATH`
in `.env` to run Core-only.

**Docker (backend only):**
```bash
docker build -f backend/Dockerfile -t sekiro-rag-backend .
docker run --rm -p 8000:8000 --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_HOST=http://host.docker.internal:11434 sekiro-rag-backend
```

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `qcwind/qwen2.5-7B-instruct-Q4_K_M` | Any instruct model works |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `VECTOR_STORE_PATH` | `./data/vector_store` | Persisted Chroma directory |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Must match the notebook's model |
| `TOP_K` | `4` | Chunks retrieved per query |
| `YOLO_MODEL_PATH` | `./data/yolo_model/best.pt` | Unset disables detection |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.5` | Measured, not a default — see above |
| `API_BASE_URL` (frontend) | `http://localhost:8000` | Read from env, never hard-coded |

---

## API reference

**`GET /health`** — readiness of each dependency; returns `200` even when detection is degraded.

**`POST /query`**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the Mortal Blade do?"}'
```
→ `{"answer": "...", "sources": [...], "detection": null}`

`422` invalid question · `502` LLM unreachable · `503` empty retrieval.

**`POST /query-image`** (Extended Track) — same as `/query`, plus a screenshot. A detection at
or above `YOLO_CONFIDENCE_THRESHOLD` filters retrieval to that boss, topping up with unfiltered
results if the filter alone would return fewer than `TOP_K` chunks. Detection is always reported
in the response, including a below-threshold miss (`used_for_retrieval: false`).

---

## Screenshots

![Empty state](docs/screenshots/01-empty-state.png)

The left rail (Conversations, Boss roster, status footer) is custom markup, not Streamlit's
sidebar — nothing about it round-trips through Python, so switching conversations never
flashes or reloads the page.

![Assistant avatar fully visible while the local index is searched](docs/screenshots/05-thinking.png)

Regenerate both with `python docs/capture_screenshots.py` (needs `playwright`, not in either
`requirements.txt` — a docs-only tool).

---

## License and attribution

Game content is © FromSoftware / Activision. Wiki text is community-authored on Fandom under
CC BY-SA. This project is an academic exercise and claims no ownership over either.
