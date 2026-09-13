"""
wiki_preprocess.py
==================
MediaWiki wikitext -> clean plain text -> section-based chunks.

The Sekiro corpus (`data/raw/raw_wiki/*.txt`) is raw Fandom/Fextralife wikitext:
navbox tables, `{{Infobox_Boss}}` / `{{Loot}}` / `{{Dialogue}}` templates,
`[[wiki links]]`, `'''bold'''` markup and `<br>`/`<hr>` HTML. Feeding that
straight into an embedder wastes tokens on markup and pollutes retrieval with
syntax noise, so it gets flattened to prose first.

Two public entry points:

    clean_wikitext(raw)                  -> str
    chunk_page(text, meta, ...)          -> list[dict]
    section_chunks / fixed_size_chunks   -> lower-level pieces

`Dialogue` templates are deliberately *kept* (rendered as readable script)
rather than dropped: direct NPC dialogue carries the obscure lore details that
Section 6's grounding questions depend on, and it appears nowhere else in the
page prose.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------
# Template / table parsing primitives
# --------------------------------------------------------------------------

# Innermost template: `{{...}}` containing no further braces. Applying this
# repeatedly bottom-up resolves arbitrarily nested templates.
_INNERMOST_TEMPLATE = re.compile(r"\{\{([^{}]*)\}\}")

# Templates dropped outright. Matched case-insensitively against the template
# name with underscores normalised to spaces.
_DROP_TEMPLATE_PREFIXES = (
    "infobox",
    "spacer",
    "clear",
    "clr",
    "navbox",
    "main",
    "see also",
    "stub",
    "quote",
    "tabber",
    "toc",
    "notebox",
    "wrongtitle",
)

# Heading patterns, e.g. `== Overview ==` / `===Phase 1===`
_HEADING = re.compile(r"^(={2,4})\s*(.+?)\s*\1\s*$", re.MULTILINE)

_WIKILINK = re.compile(r"\[\[([^\[\]|]+?)(?:\|([^\[\]]*?))?\]\]")
_EXTLINK = re.compile(r"\[(?:https?|ftp)://\S+\s+([^\]]*)\]")
_BARE_EXTLINK = re.compile(r"\[(?:https?|ftp)://\S+\]")
_FILE_LINK = re.compile(r"\[\[(?:File|Image|Media):[^\]]*\]\]", re.IGNORECASE)
_CATEGORY_LINK = re.compile(r"\[\[(?:Category|Cat):[^\]]*\]\]", re.IGNORECASE)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_REF = re.compile(r"<ref[^>/]*/?>.*?</ref>|<ref[^>]*/>", re.DOTALL | re.IGNORECASE)

_HTML_BREAK = re.compile(r"<\s*(?:br|hr)\s*/?\s*>", re.IGNORECASE)
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")

_BOLD_ITALIC = re.compile(r"'{2,}")
_MAGIC_WORD = re.compile(r"__[A-Z]+__")

# `{{PAGENAME}}` and friends expand to the page title. They must be substituted
# with the real title rather than dropped: pages are written as
# `'''{{PAGENAME}}''' is a cavernous area...`, and expanding the template to an
# empty string leaves a bare `''''''` run that partial quote-stripping turns
# into a stray apostrophe in the embedded text.
_PAGE_NAME_TEMPLATE = re.compile(
    r"\{\{\s*(?:PAGENAME|BASEPAGENAME|FULLPAGENAME|SUBPAGENAME|ARTICLEPAGENAME)\s*\}\}",
    re.IGNORECASE,
)

_ENTITIES = {
    "&nbsp;": " ",
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&#39;": "'",
    "&ndash;": "-",
    "&mdash;": "-",
    "&hellip;": "...",
    "&times;": "x",
}


def _split_top_level(body: str, sep: str = "|") -> list[str]:
    """Split on `sep` at brace/bracket depth 0.

    A naive `body.split('|')` destroys templates whose parameters contain
    nested templates or wiki links with pipes in them (e.g. `[[Page|Label]]`),
    so nesting depth is tracked explicitly.
    """
    parts, buf, depth_brace, depth_bracket = [], [], 0, 0
    i = 0
    while i < len(body):
        two = body[i : i + 2]
        if two == "{{":
            depth_brace += 1
            buf.append(two)
            i += 2
            continue
        if two == "}}":
            depth_brace -= 1
            buf.append(two)
            i += 2
            continue
        if two == "[[":
            depth_bracket += 1
            buf.append(two)
            i += 2
            continue
        if two == "]]":
            depth_bracket -= 1
            buf.append(two)
            i += 2
            continue
        ch = body[i]
        if ch == sep and depth_brace == 0 and depth_bracket == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _parse_template(body: str) -> tuple[str, list[str], list[tuple[str, str]]]:
    """Split a template body into (name, positional_args, named_args)."""
    pieces = _split_top_level(body)
    name = pieces[0].strip()
    positional: list[str] = []
    named: list[tuple[str, str]] = []
    for piece in pieces[1:]:
        key, sep, value = piece.partition("=")
        # A `=` inside a nested link/template is not a parameter assignment.
        if sep and "[" not in key and "{" not in key and key.strip():
            named.append((key.strip(), value.strip()))
        else:
            positional.append(piece.strip())
    return name, positional, named


def _trailing_int(key: str) -> int:
    match = re.search(r"(\d+)\s*$", key)
    return int(match.group(1)) if match else 0


def _render_dialogue(named: list[tuple[str, str]]) -> str:
    """Render `{{Dialogue|Event1=..|Dialogue1=..}}` as readable script."""
    events, lines = {}, {}
    for key, value in named:
        low = key.lower().replace(" ", "")
        if low.startswith("event"):
            events[_trailing_int(key)] = value
        elif low.startswith("dialogue"):
            lines[_trailing_int(key)] = value

    out: list[str] = []
    for idx in sorted(lines):
        text = lines[idx]
        if not text:
            continue
        event = events.get(idx)
        if event:
            out.append(f"[{event}]")
        out.append(text)
    return "\n".join(out)


def _render_loot(named: list[tuple[str, str]]) -> str:
    """Render `{{Loot|XP=..|Loot1_Name=..}}` as a prose sentence."""
    names, scalars = [], []
    for key, value in named:
        low = key.lower()
        if not value:
            continue
        if low.endswith("name"):
            names.append(value)
        elif low in ("xp", "currencyamount"):
            scalars.append(f"{key}={value}")
    if not names and not scalars:
        return ""
    text = "Rewards: " + ", ".join(names) if names else ""
    if scalars:
        text += (" " if text else "") + "(" + ", ".join(scalars) + ")"
    return text.strip()


def _render_template(body: str) -> str:
    """Turn one template body into plain text (possibly empty)."""
    name, positional, named = _parse_template(body)
    norm = name.lower().replace("_", " ").strip()

    if any(norm.startswith(prefix) for prefix in _DROP_TEMPLATE_PREFIXES):
        return ""
    if norm.startswith("dialogue"):
        return _render_dialogue(named)
    if norm.startswith("loot"):
        return _render_loot(named)
    if norm.startswith("header"):
        # `{{Header|HeaderType=3|HeaderText=Ending 2: Immortal Severance}}`
        for key, value in named:
            if key.lower().replace("_", "").endswith("headertext"):
                return value
        return ""

    # Generic case: a template wrapping a single unnamed string is really just
    # inline emphasis/annotation (`{{PoisonAttack|Owl's Flight - Poison}}`), so
    # keep that string. Anything else is structural noise -> drop.
    if len(positional) == 1 and not named:
        return positional[0]
    if positional:
        return " ".join(positional)
    return ""


def _expand_templates(text: str) -> str:
    """Resolve all templates bottom-up, then drop any unmatched braces."""
    for _ in range(50):  # depth guard; real nesting is < 5
        expanded = _INNERMOST_TEMPLATE.sub(lambda m: _render_template(m.group(1)), text)
        if expanded == text:
            break
        text = expanded
    return re.sub(r"\{\{|\}\}", " ", text)


# Table syntax. Cell *contents* are kept — see `_unwrap_tables`.
_TABLE_OPEN = re.compile(r"^[ \t]*\{\|.*$", re.MULTILINE)
_TABLE_CLOSE = re.compile(r"^[ \t]*\|\}.*$", re.MULTILINE)
_TABLE_ROW = re.compile(r"^[ \t]*\|[-+].*$", re.MULTILINE)
# Leading `colspan="2" style="..." |` cell-attribute prefix, which is never
# page content. Deliberately excludes `[ ] { }` so wiki links and templates
# containing `=` are not mistaken for attributes.
_ATTR_PREFIX = re.compile(r"^[^|\[\]{}]*=[^|]*\|")


def _unwrap_tables(text: str) -> str:
    """Strip table *structure*, keeping cell contents.

    Tables in this corpus are used two ways, and both need to keep their text:
    as navboxes (a row of sibling page names) and — critically — as *layout*
    wrappers. The four Ending pages put their entire body inside a single
    `{| ... |}`, so deleting table blocks wholesale erases whole pages that
    the evaluation question set depends on. Unwrapping instead removes only
    the syntax (`{|`, `|-`, cell attributes) and leaves the prose.
    """
    text = _TABLE_OPEN.sub("", text)
    text = _TABLE_CLOSE.sub("", text)
    text = _TABLE_ROW.sub("", text)

    out: list[str] = []
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("||"):
            stripped = stripped[2:]
        elif stripped.startswith("|"):
            stripped = stripped[1:]
        else:
            out.append(line)
            continue

        stripped = stripped.strip()
        attr_match = _ATTR_PREFIX.match(stripped)
        if attr_match:
            stripped = stripped[attr_match.end() :]
        out.append(stripped.strip())
    return "\n".join(out)


_REDIRECT = re.compile(r"^\s*#\s*redirect\b", re.IGNORECASE | re.MULTILINE)


def is_redirect(raw: str) -> bool:
    """True for alias pages (`#REDIRECT [[Target]]`), which carry no content.

    These would otherwise become stub chunks like
    "Shugendo / REDIRECT Senpou Temple#Temple Grounds" that can win retrieval
    slots while contributing nothing.
    """
    return bool(_REDIRECT.match(raw.strip()))


def _strip_links(text: str) -> str:
    text = _FILE_LINK.sub(" ", text)
    text = _CATEGORY_LINK.sub(" ", text)
    # [[Target|Label]] -> Label ; [[Target]] -> Target
    text = _WIKILINK.sub(lambda m: (m.group(2) or m.group(1)).strip(), text)
    text = _EXTLINK.sub(lambda m: m.group(1).strip(), text)
    text = _BARE_EXTLINK.sub(" ", text)
    return text


def _strip_emphasis(text: str) -> str:
    """Remove wiki quote-markup runs (`''italic''`, `'''bold'''`).

    Runs of 2+ apostrophes are markup and are removed outright. Deliberately
    *not* done by unwrapping `'''x'''` -> `x`: some navboxes open a bold
    wikilink whose display text itself begins with `'''`, producing an
    adjacent six-quote run that a `{3,5}` unwrap consumes only partially and
    leaves an orphan apostrophe in the embedded text. Removing the runs
    directly is immune to that, and leaves single apostrophes — contractions
    and possessives like "Owl's" / "merchants'" — untouched.
    """
    return _BOLD_ITALIC.sub("", text)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_wikitext(raw: str, page_name: str = "") -> str:
    """Flatten one raw wikitext page into clean prose, preserving headings.

    `page_name` supplies the value for `{{PAGENAME}}`-style templates; pass the
    page title (usually the filename with underscores replaced) or those
    templates expand to nothing and leave orphaned quote markup behind.
    """
    text = raw
    text = _COMMENT.sub(" ", text)
    text = _REF.sub(" ", text)
    if page_name:
        text = _PAGE_NAME_TEMPLATE.sub(lambda _: page_name, text)
    else:
        text = _PAGE_NAME_TEMPLATE.sub(" ", text)
    # Templates before tables: a template body can span newlines and contain
    # `|`, which the line-oriented table pass would otherwise mistake for
    # cell syntax and corrupt.
    text = _expand_templates(text)
    text = _unwrap_tables(text)
    text = _HTML_BREAK.sub("\n", text)
    text = _HTML_TAG.sub("", text)
    text = _strip_links(text)
    text = _strip_emphasis(text)
    text = _MAGIC_WORD.sub(" ", text)
    for entity, char in _ENTITIES.items():
        text = text.replace(entity, char)
    # Drop bullet/indent markers, keeping the text they prefix.
    text = re.sub(r"^[*:#]+\s*", "", text, flags=re.MULTILINE)
    return _normalize_whitespace(text)


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

# Rough token estimate: MiniLM's WordPiece tokenizer averages ~1.3 tokens per
# whitespace-separated word on this corpus, plus ~1 token for the separator.
_TOKENS_PER_WORD = 1.3


def estimate_tokens(text: str) -> int:
    """Cheap token estimate, avoiding a tokenizer dependency at chunk time."""
    return int(len(text.split()) * _TOKENS_PER_WORD)


def _split_oversized(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Fixed-size word window over a section that exceeds `max_tokens`."""
    words = text.split()
    if not words:
        return []

    max_words = max(1, int(max_tokens / _TOKENS_PER_WORD))
    overlap_words = max(0, min(int(overlap_tokens / _TOKENS_PER_WORD), max_words - 1))
    step = max(1, max_words - overlap_words)

    pieces = []
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        if not window:
            break
        pieces.append(" ".join(window))
        if start + max_words >= len(words):
            break
    return pieces


def _split_to_pieces(
    body: str, max_tokens: int, overlap_tokens: int, min_tokens: int
) -> list[str]:
    """Split one section body, merging slivers into their predecessor."""
    if estimate_tokens(body) <= max_tokens:
        return [body]

    pieces: list[str] = []
    for piece in _split_oversized(body, max_tokens, overlap_tokens):
        # Merge a too-small tail into the previous piece rather than emitting a
        # context-free fragment.
        if pieces and estimate_tokens(piece) < min_tokens:
            pieces[-1] = pieces[-1] + "\n" + piece
        else:
            pieces.append(piece)
    return pieces


def _iter_sections(text: str) -> list[tuple[str, str]]:
    """Yield `(heading_path, body)` pairs split on `==`/`===`/`====` headings."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("", text.strip())] if text.strip() else []

    sections: list[tuple[str, str]] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    # Heading stack, so `===Phase 1===` under `==Behaviour and Tactics==`
    # becomes "Behaviour and Tactics > Phase 1" and a retrieved chunk stays
    # self-describing. Popping stale entries at each level prevents a previous
    # sibling section's heading leaking into the breadcrumb.
    stack: list[tuple[int, str]] = []

    for idx, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        path = " > ".join(t for _, t in stack)
        sections.append((path, body))

    return sections


def chunk_page(
    text: str,
    source: str,
    boss: str = "",
    max_tokens: int = 400,
    overlap_tokens: int = 75,
    min_tokens: int = 40,
) -> list[dict]:
    """Split one cleaned page into section-based chunks.

    Strategy (Section 5.2 of the spec):
      * Primary: split on the page's own `==`/`===` subheadings, since these
        sources are already authored as Overview / Strategy / Lore / Dialogue.
      * Oversized sections are further windowed at `max_tokens` with
        `overlap_tokens` carried over, so a long "Behaviour and Tactics"
        section still yields several focused chunks.
      * Pages with no headings fall back entirely to fixed-size chunking.
      * Slivers under `min_tokens` merge into the preceding chunk.

    `boss` is the YOLO class name to tag this page's chunks with (Section
    5.5.8). Chroma metadata cannot store `None`, so non-boss pages use `""`.
    """
    sections = _iter_sections(text)

    # A short leading preamble is usually the page's navbox: a "•"-separated
    # row of sibling page names. Left alone it becomes a junk chunk that can
    # occupy a retrieval slot while contributing nothing. Folded into the next
    # real section it turns into useful alias text ("Owl", "Owl (Father)")
    # attached to content that actually answers questions.
    if (
        len(sections) > 1
        and sections[0][0] == ""
        and estimate_tokens(sections[0][1]) < min_tokens
    ):
        head = sections[0][1]
        sections = [(sections[1][0], f"{head}\n{sections[1][1]}")] + sections[2:]

    chunks: list[dict] = []

    for path, body in sections:
        if not body:
            continue
        # `_iter_sections` returns the whole page as one unnamed section when
        # there are no headings; that is the fixed-size fallback path.
        for piece in _split_to_pieces(body, max_tokens, overlap_tokens, min_tokens):
            if estimate_tokens(piece) < min_tokens and chunks:
                chunks[-1]["text"] += "\n\n" + (
                    f"{path}\n{piece}" if path else piece
                )
                chunks[-1]["token_estimate"] = estimate_tokens(chunks[-1]["text"])
                continue

            prefix = f"{source} — {path}" if path else source
            chunks.append(
                {
                    "text": f"{prefix}\n{piece}",
                    "source": source,
                    "section": path,
                    "boss": boss,
                    "token_estimate": estimate_tokens(piece),
                }
            )

    return chunks
