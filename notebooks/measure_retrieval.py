"""Throwaway: measure retrieval hit-rate over the 10 spec evaluation questions.

Prints, for each question, the top-4 retrieved pages and whether the gold page is
in that window -- the window is what actually reaches the LLM, so it is the right
thing to score. Not part of the deliverable.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chromadb
from sentence_transformers import SentenceTransformer

STORE = Path(__file__).resolve().parent / "vector_store_export"
TOP_K = 4

QUESTIONS = [
    ("Q1", "well-known", "Who is Genichiro Ashina and what is his role in the story?", ["Genichiro"]),
    ("Q2", "well-known", "What is the Dragon's Heritage, and who carries it?", []),
    ("Q3", "well-known", "How do you defeat the Guardian Ape?", ["Guardian Ape"]),
    ("Q4", "well-known", "What happens in the Immortal Severance ending?", ["Ending 2- Immortal Severance"]),
    ("Q5", "obscure", "What is Kuro's connection to the Dragon's Heritage?", ["Kuro"]),
    ("Q6", "well-known", "What does the Mortal Blade do?", ["Mortal Blade"]),
    ("Q7", "obscure", "Who is the Sculptor, and what is his backstory?", ["Sculptor"]),
    ("Q8", "well-known", "What is the difference between the Shura ending and the Return ending?", ["Ending 1- Shura", "Ending 4- Return"]),
    ("Q9", "well-known", "What prosthetic tool is effective against Lady Butterfly?", ["Lady Butterfly"]),
    ("Q10", "obscure", "What is Emma's role in Genichiro's story?", ["Emma"]),
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=str(STORE))
col = client.get_collection("sekiro_wiki")

hits = 0
detailed = []
for qid, kind, question, gold in QUESTIONS:
    emb = embedder.encode([question], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=emb, n_results=TOP_K)
    meta = res["metadatas"][0]
    pages = []
    for m in meta:
        page = m["source"].replace("_", " ")
        if page not in pages:
            pages.append(page)
    if not gold:
        verdict = "CORPUS GAP"
    elif all(any(g.lower() in p.lower() for p in pages) for g in gold):
        verdict = "HIT"
        hits += 1
    else:
        verdict = "MISS"
    detailed.append((qid, kind, question, pages, verdict))
    print(f"{qid} [{kind}] {verdict}")
    print(f"    {question}")
    for p in pages:
        print(f"      - {p}")
    print()

scored = [d for d in detailed if d[4] != "CORPUS GAP"]
print(f"Top-{TOP_K} page coverage: {hits}/{len(scored)} scoreable questions")
print(f"Unique pages across all windows: {len({p for d in detailed for p in d[3]})}")

json.dump([{"qid": d[0], "kind": d[1], "question": d[2], "pages": d[3], "verdict": d[4]} for d in detailed],
          open("eval_retrieval.json", "w", encoding="utf-8"), indent=2)
print("wrote eval_retrieval.json")
