"""Throwaway: regenerate CORPUS.md (indexed pages + source URLs).

Reads the persisted index so the list reflects what was actually embedded rather
than what was on disk. Not part of the deliverable.
"""
from collections import Counter
from pathlib import Path

import chromadb

BASE = Path(__file__).resolve().parent.parent  # repo root (this file lives in notebooks/)
WIKI = "https://sekiro-shadows-die-twice.fandom.com/wiki/"

client = chromadb.PersistentClient(path=str(BASE / "notebooks" / "vector_store_export"))
col = client.get_collection("sekiro_wiki")
metas = col.get(include=["metadatas"])["metadatas"]

per_page: Counter[str] = Counter()
boss_of: dict[str, str] = {}
for m in metas:
    per_page[m["source"]] += 1
    if m.get("boss"):
        boss_of[m["source"]] = m["boss"]

raw_stems = {p.stem.replace("_", " ") for p in (BASE / "data" / "raw" / "raw_wiki").glob("*.txt")}
skipped = raw_stems - set(per_page)

out: list[str] = []
out.append("# Corpus: pages used")
out.append("")
out.append(
    f"{len(per_page)} pages were indexed into the vector store, contributing "
    f"{sum(per_page.values())} chunks. The corpus was scraped as raw MediaWiki source from the "
    f"[Sekiro: Shadows Die Twice Fandom wiki]"
    f"(https://sekiro-shadows-die-twice.fandom.com/wiki/Sekiro:_Shadows_Die_Twice_Wiki)."
)
out.append("")
out.append("Re-fetch any page by appending `?action=raw` to its URL:")
out.append("")
out.append("```")
out.append(WIKI + "<Page_Name>?action=raw")
out.append("```")
out.append("")
out.append(
    f"The scrape contained {len(raw_stems)} files in total; the remaining {len(skipped)} are "
    "redirect stubs (alias pages such as `Shugendo` pointing at a section of another page). "
    "They carry no prose of their own and are skipped during loading, which is why the chunk "
    "and page counts differ."
)
out.append("")
out.append("## Indexed pages")
out.append("")
out.append("| Page | Chunks | Boss class |")
out.append("|---|---|---|")
for page, n in sorted(per_page.items()):
    url = WIKI + page.replace(" ", "_")
    out.append(f"| [`{page}`]({url}) | {n} | {boss_of.get(page, '—')} |")
out.append("")

(BASE / "CORPUS.md").write_text("\n".join(out), encoding="utf-8")
print(f"indexed pages : {len(per_page)}")
print(f"total chunks  : {sum(per_page.values())}")
print(f"boss pages    : {len(boss_of)}")
print(f"skipped stubs : {len(skipped)}")
print("wrote CORPUS.md")
