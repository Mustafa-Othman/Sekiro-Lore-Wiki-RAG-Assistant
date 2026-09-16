"""Sekiro Geeker — Streamlit chat UI for the wiki RAG assistant.

Run with:

    cd frontend && streamlit run app.py
"""

from __future__ import annotations

import base64
import re
import uuid
from html import escape
from pathlib import Path
from urllib.parse import quote

import streamlit as st
import streamlit.components.v1 as components

import api_client

ASSETS = Path(__file__).resolve().parent / "assets"
AVATAR_USER = str(ASSETS / "avatar_user.svg")
AVATAR_ASSISTANT = str(ASSETS / "avatar_assistant.png")
BACKDROP_PATH = ASSETS / "welcome_backdrop.png"
LOGO_PATH = ASSETS / "logo_header.png"

INPUT_BAR_RESERVE_PX = 88

# YOLO class labels (display names from the 40-class dataset)
BOSS_CLASSES = [
    "Armored Warrior",
    "Ashina Elite - Jinsuke Saze",
    "Ashina Elite - Ujinari Mizuo",
    "Blazing Bull",
    "Chained Ogre",
    "Corrupted Monk",
    "Demon of Hatred",
    "Divine Dragon",
    "Folding Screen Monkeys",
    "General Kuranosuke Matsumoto",
    "General Naomori Kawarada",
    "General Tenzen Yamauchi",
    "Genichiro Phase 1",
    "Genichiro Phase 2",
    "Guardian Ape",
    "Gyoubu Oniwa",
    "Headless",
    "Isshin - the Sword Saint",
    "Juzou the Drunkard",
    "Lady Butterfly",
    "Leader Shigenori Yamauchi",
    "Lone Shadow Longswordsman",
    "Lone Shadow Masanaga the Spear-Bearer",
    "Lone Shadow Vilehand",
    "Long-arm Centipede Giraffe",
    "Long-arm Centipede Senun",
    "Mist Noble",
    "ORin of the Water",
    "Okami Leader Shizu",
    "Old Dragons of the Tree",
    "Owl",
    "Sakura Bull of the Palace",
    "Seven Ashina Spears - Shikibu Toshikatsu Yamauchi",
    "Seven Ashina Spears - Shume Masaji Oniwa",
    "Shichimen Warrior",
    "Shigekichi of the Red Guard",
    "Shinobi Hunter Enshin of Misen",
    "Snake Eyes Shirafuji",
    "Snake Eyes Shirahagi",
    "Tokujiro the Glutton",
]

# Ink-brush edge (faint irregular stroke — not a hard border)
INK_BRUSH_SVG = "data:image/svg+xml," + quote(
    """<svg xmlns='http://www.w3.org/2000/svg' width='18' height='800' viewBox='0 0 18 800' preserveAspectRatio='none'>
  <path d='M9 0c1.2 40 .2 80 1.4 120-.8 55 1.6 90-.4 140 1.1 48-.6 95 1.2 150-.9 60 1.4 110-.3 170 1 55-.7 110 .6 160-.5 20 .4 40-.2 60'
        fill='none' stroke='#8B93A6' stroke-width='2.2' stroke-linecap='round' opacity='0.45'/>
  <path d='M7.5 20c.9 50-.4 95 1 150-1 70 1.2 120-.2 185.8 55-1 100 1.1 160-.8 70 1.3 130-.4 190'
        fill='none' stroke='#8B93A6' stroke-width='1.1' stroke-linecap='round' opacity='0.28'/>
</svg>"""
)

st.set_page_config(
    page_title="Sekiro Geeker",
    page_icon=AVATAR_ASSISTANT,
    layout="centered",
    initial_sidebar_state="collapsed",
)

EXAMPLE_QUESTIONS = [
    "What is the Mortal Blade and what does it do?",
    "Who is the Sculptor, and what is his backstory?",
    "What are the requirements for the Purification ending?",
]

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Spectral:wght@600;700&display=swap');

:root {{
  --bg: #14171F;
  --panel: #1B1F2A;
  --text: #E8E2D4;
  --muted: #8B93A6;
  --accent: #C89B3C;
  --error: #7A2331;
  --line: rgba(232, 226, 212, 0.10);
  --col: 760px;
  --drawer: 300px;
  --input-reserve: {INPUT_BAR_RESERVE_PX}px;
}}

@media (min-width: 1400px) {{
  :root {{ --col: 920px; }}
}}

/* A permanently-open rail has to give up some room on small screens or the
   chat column would be squeezed to nothing. */
@media (max-width: 900px) {{
  :root {{ --drawer: min(210px, 54vw); }}
}}

html, body, [class*="css"], .stApp, .stMarkdown, label, span {{
  font-family: "IBM Plex Sans", sans-serif !important;
  color: var(--text);
}}

.stApp {{ background: var(--bg) !important; }}

#MainMenu,
footer,
header[data-testid="stHeader"],
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
section[data-testid="stSidebar"] {{
  display: none !important;
  visibility: hidden !important;
}}

/* Permanent left rail. There is no toggle and no open/closed state at all, so
   nothing has to round-trip through Python -- the panel simply is always
   there, and a rerun cannot flash it shut. */
.sg-drawer {{
  position: fixed;
  top: 0;
  left: 0;
  z-index: 1000001;
  width: var(--drawer);
  height: 100vh;
  background: #1B1F2A;
  padding: 1.15rem 1.05rem 1.35rem;
  box-sizing: border-box;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}}

/* Ink-brush edge on the drawer */
.sg-drawer::after {{
  content: "";
  position: absolute;
  top: 0;
  right: -9px;
  width: 18px;
  height: 100%;
  background: url("{INK_BRUSH_SVG}") center / 18px 100% no-repeat;
  opacity: 0.85;
  pointer-events: none;
}}

/* The rail is fixed, so its width has to be reserved on the app container --
   otherwise the centred chat column and the bottom dock slide under it. */
[data-testid="stAppViewContainer"] {{
  padding-left: var(--drawer) !important;
  box-sizing: border-box !important;
}}

.sg-side-kicker {{
  font-family: "Spectral", serif !important;
  font-size: 0.72rem !important;
  color: var(--muted) !important;
  letter-spacing: 0.02em;
  margin: 0 0 0.55rem !important;
}}

.sg-side-gap {{ height: 1.35rem; }}

.sg-new {{
  display: inline-block;
  color: var(--text);
  font-size: 0.88rem;
  font-weight: 500;
  text-decoration: none !important;
  margin: 0 0 0.65rem;
  padding: 0.15rem 0;
}}

.sg-new:hover {{ color: var(--accent); }}

.sg-hist {{
  display: block;
  position: relative;
  padding: 0.42rem 0.55rem 0.42rem 0.7rem;
  margin: 0.1rem 0;
  color: var(--muted);
  font-size: 0.82rem;
  line-height: 1.35;
  text-decoration: none !important;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  border-left: 2px solid transparent;
  transition: background 0.15s ease, color 0.15s ease;
}}

.sg-hist:hover {{
  background: rgba(255, 255, 255, 0.08);
  color: var(--text);
}}

.sg-hist.is-active {{
  color: var(--text);
  border-left-color: var(--accent);
}}

.sg-hist-empty {{
  color: var(--muted);
  font-size: 0.78rem;
  margin: 0.35rem 0 0;
}}

.sg-roster {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.65rem 0.55rem;
  margin: 0.15rem 0 0;
}}

.sg-boss {{
  width: 52px;
  text-align: center;
}}

.sg-boss-orb {{
  width: 36px;
  height: 36px;
  margin: 0 auto 0.28rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.68rem;
  font-weight: 600;
  color: var(--text);
  background:
    radial-gradient(circle at 35% 30%, rgba(232,226,212,0.16), transparent 55%),
    #14171F;
  border: 1px solid rgba(139, 147, 166, 0.45);
  box-sizing: border-box;
}}

.sg-boss.is-hot .sg-boss-orb {{
  border: 2px solid var(--accent);
  box-shadow: 0 0 0 1px rgba(200, 155, 60, 0.25);
}}

.sg-boss-name {{
  display: block;
  font-size: 0.58rem;
  line-height: 1.2;
  color: var(--muted);
  max-width: 52px;
  margin: 0 auto;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}

.sg-side-foot {{
  margin-top: auto;
  padding-top: 1.5rem;
}}

.sg-side-foot-line {{
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.75rem;
  color: var(--muted);
  margin: 0;
}}

.sg-side-foot-meta {{
  font-size: 0.68rem;
  color: var(--muted);
  opacity: 0.85;
  margin: 0.28rem 0 0;
  line-height: 1.35;
}}

.sg-dot {{
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--muted);
  flex-shrink: 0;
}}
.sg-dot.ok {{ background: #5a8f6e; }}
.sg-dot.warn {{ background: #c9a227; }}
.sg-dot.bad {{ background: var(--error); }}

/* ---------- Main chrome (existing) ---------- */
.sg-backdrop {{
  position: fixed !important;
  inset: 0 !important;
  z-index: 0 !important;
  pointer-events: none !important;
  background-repeat: no-repeat !important;
  background-position: center 20% !important;
  background-size: cover !important;
  opacity: 1 !important;
  -webkit-mask-image: radial-gradient(ellipse 110% 95% at 50% 42%,
    #000 0%, #000 62%, rgba(0,0,0,0.65) 82%, rgba(0,0,0,0.25) 94%, transparent 100%);
  mask-image: radial-gradient(ellipse 110% 95% at 50% 42%,
    #000 0%, #000 62%, rgba(0,0,0,0.65) 82%, rgba(0,0,0,0.25) 94%, transparent 100%);
}}

.sg-scrim {{
  position: fixed !important;
  inset: 0 !important;
  z-index: 0 !important;
  pointer-events: none !important;
  background: radial-gradient(
    ellipse 58% 78% at 50% 42%,
    rgba(20, 23, 31, 0.82) 0%,
    rgba(20, 23, 31, 0.62) 42%,
    rgba(20, 23, 31, 0.32) 70%,
    rgba(20, 23, 31, 0.14) 100%
  ) !important;
}}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main,
[data-testid="stMainBlockContainer"],
.block-container,
[data-testid="stBottom"] {{
  position: relative;
  z-index: 1;
}}

[data-testid="stMainBlockContainer"],
.block-container {{
  max-width: var(--col) !important;
  width: min(100%, var(--col)) !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding-left: 1rem !important;
  padding-right: 1rem !important;
  padding-bottom: 5.25rem !important;
}}

body:has(.sg-welcome-marker) [data-testid="stMainBlockContainer"],
body:has(.sg-welcome-marker) .block-container {{
  height: calc(100vh - var(--input-reserve)) !important;
  min-height: calc(100vh - var(--input-reserve)) !important;
  max-height: calc(100vh - var(--input-reserve)) !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
  padding-top: 0 !important;
  box-sizing: border-box !important;
  overflow: hidden !important;
}}

body:has(.sg-welcome-marker) [data-testid="stMainBlockContainer"] > div,
body:has(.sg-welcome-marker) .block-container > div {{
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
}}

.sg-welcome-marker {{
  display: block !important;
  height: 0 !important;
  width: 0 !important;
  overflow: hidden !important;
  margin: 0 !important;
  padding: 0 !important;
}}

.sg-header {{ position: relative; margin: 0 0 0.85rem; z-index: 1; }}

.sg-brand-row {{
  display: flex !important;
  align-items: center !important;
  gap: 0.95rem !important;
}}

img.sg-logo,
.sg-logo,
.stMarkdown img.sg-logo,
[data-testid="stMarkdownContainer"] img.sg-logo {{
  width: 64px !important;
  height: 64px !important;
  min-width: 64px !important;
  min-height: 64px !important;
  max-width: 64px !important;
  max-height: 64px !important;
  border-radius: 14px !important;
  object-fit: cover !important;
  flex-shrink: 0 !important;
  display: block !important;
  border: 1px solid var(--line) !important;
}}

p.sg-wordmark,
.sg-wordmark,
.stMarkdown p.sg-wordmark {{
  font-family: "Spectral", "Times New Roman", serif !important;
  font-weight: 700 !important;
  font-size: 36px !important;
  line-height: 1.05 !important;
  letter-spacing: -0.02em !important;
  color: var(--text) !important;
  margin: 0 !important;
}}

p.sg-tagline,
.sg-tagline {{
  font-family: "IBM Plex Sans", sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 400 !important;
  line-height: 1.45 !important;
  color: var(--muted) !important;
  margin: 0.55rem 0 0 !important;
  max-width: 36rem;
}}

.sg-empty-hint {{
  color: var(--muted);
  font-size: 0.8rem;
  margin: 1.1rem 0 0.55rem;
}}

div[data-testid="stButton"] > button {{
  background: rgba(27, 31, 42, 0.92) !important;
  color: var(--text) !important;
  border: 1px solid var(--line) !important;
  border-radius: 999px !important;
  box-shadow: none !important;
  font-family: "IBM Plex Sans", sans-serif !important;
  font-size: 0.8rem !important;
  font-weight: 450 !important;
  padding: 0.4rem 0.8rem !important;
  min-height: 0 !important;
  white-space: normal !important;
  text-align: left !important;
  line-height: 1.35 !important;
}}

div[data-testid="stButton"] > button:hover {{
  border-color: rgba(200, 155, 60, 0.55) !important;
}}

[data-testid="stChatMessage"] {{
  background: rgba(27, 31, 42, 0.94) !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  padding: 0.7rem 0.85rem !important;
  margin-bottom: 0.6rem !important;
  box-shadow: none !important;
  box-sizing: border-box !important;
  overflow: hidden !important;
}}

[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {{
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  box-sizing: border-box !important;
}}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {{
  color: var(--text) !important;
  font-size: 0.95rem;
  line-height: 1.6;
}}

[data-testid="stChatMessageAvatarImage"],
[data-testid="stChatMessage"] img[alt="assistant avatar"],
[data-testid="stChatMessage"] img[alt="user avatar"],
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] img,
[data-testid="stChatMessage"] img {{
  width: 38px !important;
  height: 38px !important;
  min-width: 38px !important;
  min-height: 38px !important;
  border-radius: 50% !important;
  object-fit: cover !important;
  opacity: 1 !important;
  filter: none !important;
  visibility: visible !important;
}}

/* While a turn is in flight Streamlit marks the still-unrefreshed elements
   "stale" and fades them (opacity -> theme.stale) with a 1s transition. Keep
   that fade off the avatar and its wrappers, so the icon reads the same while
   the answer is being fetched as it does once the answer lands. */
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"],
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarImage"],
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] {{
  opacity: 1 !important;
  transition: none !important;
}}

/* The assistant portrait is a very dark piece of art (brightest pixel ~87/255)
   on a near-black card, so at 38px it read as a faded smudge. Lift it and give
   it a hairline rim so it is legible, loading or idle. */
[data-testid="stChatMessage"] img[alt="assistant avatar"] {{
  filter: brightness(1.62) contrast(1.06) !important;
  box-shadow: 0 0 0 1px rgba(232, 226, 212, 0.22);
}}

/* Loading cue: the rim breathes gold instead of the icon fading out. Opacity
   stays pinned at 1 (the !important rule above beats animation keyframes
   anyway), so the avatar can never dip below full strength. */
[data-testid="stChatMessage"]:has(.sg-searching) img[alt="assistant avatar"] {{
  animation: sg-avatar-breathe 1.9s ease-in-out infinite;
}}

@keyframes sg-avatar-breathe {{
  0%, 100% {{ box-shadow: 0 0 0 1px rgba(232, 226, 212, 0.22); }}
  50%      {{ box-shadow: 0 0 0 2px rgba(200, 155, 60, 0.62); }}
}}

[data-testid="stChatMessage"] [data-testid="stSpinner"],
[data-testid="stChatMessage"] .stSpinner {{
  display: none !important;
}}

.sg-searching {{
  color: var(--muted) !important;
  font-size: 0.9rem !important;
  margin: 0 !important;
}}

.sg-detect {{
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  margin: 0 0 0.65rem;
  padding: 0.22rem 0.65rem;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: rgba(20, 23, 31, 0.7);
  color: var(--muted);
  font-size: 0.75rem;
  font-weight: 500;
}}

.sg-detect strong {{ color: var(--text); font-weight: 600; }}
.sg-detect svg {{
  width: 12px; height: 12px; stroke: var(--muted); fill: none; flex-shrink: 0;
}}

.sg-sources-wrap {{
  display: block !important;
  width: 100% !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
  margin: 0.65rem 0 0 !important;
  padding: 0 !important;
}}

.sg-sources {{
  display: block !important;
  box-sizing: border-box !important;
  width: 100% !important;
  max-width: 100% !important;
  margin: 0 !important;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(20, 23, 31, 0.45);
  padding: 0.15rem 0.7rem 0.35rem;
}}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"]:has(.sg-sources),
[data-testid="stChatMessage"] [data-testid="stElementContainer"]:has(.sg-sources) {{
  width: 100% !important;
  max-width: 100% !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
  margin-left: 0 !important;
  margin-right: 0 !important;
}}

.sg-sources > summary {{
  list-style: none;
  cursor: pointer;
  color: var(--muted);
  font-size: 0.8rem;
  font-weight: 500;
  padding: 0.4rem 0;
  user-select: none;
}}

.sg-sources > summary::-webkit-details-marker {{ display: none; }}
.sg-sources > summary::marker {{ content: ""; }}
.sg-sources > summary .sg-chevron {{
  display: inline-block;
  width: 0.9rem;
  color: var(--muted);
  font-weight: 600;
}}
.sg-sources[open] > summary .sg-chevron {{ transform: rotate(90deg); }}
.sg-sources ul {{
  margin: 0 0 0.35rem;
  padding-left: 1.1rem;
  color: var(--text);
  font-size: 0.85rem;
}}
.sg-sources code {{
  color: var(--text);
  background: rgba(20, 23, 31, 0.65);
  padding: 0.05rem 0.3rem;
  border-radius: 4px;
  font-size: 0.8rem;
}}

[data-testid="stAlert"] {{
  background: var(--error) !important;
  color: var(--text) !important;
  border: none !important;
}}

/* Dock is full-bleed but transparent; the dark panel lives on the inner block
   so it lines up with the chat column above instead of running edge-to-edge. */
[data-testid="stBottom"] {{
  background: transparent !important;
  border-top: none !important;
  display: flex !important;
  justify-content: center !important;
  z-index: 2 !important;
}}

[data-testid="stBottom"] > div {{
  /* --col minus the block container's 1rem side padding == the chat column,
     so the panel's edges sit flush under the message cards. Streamlit pins
     min-width:100% on this node, which would beat max-width -- release it. */
  min-width: 0 !important;
  max-width: calc(var(--col) - 2rem) !important;
  width: min(calc(100% - 2rem), calc(var(--col) - 2rem)) !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding: 0.5rem 0.6rem 0.65rem !important;
  box-sizing: border-box !important;
  background: rgba(20, 23, 31, 0.92) !important;
  border: 1px solid var(--line) !important;
  border-bottom: none !important;
  border-radius: 14px 14px 0 0 !important;
  backdrop-filter: blur(6px);
}}

[data-testid="stChatInput"] {{ background: transparent !important; }}
[data-testid="stChatInput"] > div {{
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
  border-radius: 999px !important;
  box-shadow: none !important;
  padding: 0.1rem 0.2rem !important;
}}
[data-testid="stChatInput"] textarea {{
  font-family: "IBM Plex Sans", sans-serif !important;
  color: var(--text) !important;
  caret-color: var(--accent) !important;
  background: transparent !important;
  font-size: 0.92rem !important;
  padding-top: 0.55rem !important;
  padding-bottom: 0.55rem !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{
  color: var(--muted) !important;
  opacity: 1 !important;
}}
[data-testid="stChatInputSubmitButton"] {{
  background: var(--accent) !important;
  color: #14171F !important;
  border: none !important;
  border-radius: 999px !important;
  width: 2rem !important;
  height: 2rem !important;
  min-height: 2rem !important;
  min-width: 2rem !important;
}}
[data-testid="stChatInputSubmitButton"] svg {{
  fill: #14171F !important;
  stroke: #14171F !important;
}}
[data-testid="stChatInputFileUploadButton"],
[data-testid="stChatInput"] button[kind="secondary"] {{
  color: var(--muted) !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}}

.sg-embers {{
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  overflow: hidden;
}}
.sg-ember {{
  position: absolute;
  width: 3px;
  height: 3px;
  border-radius: 50%;
  background: rgba(200, 155, 60, 0.35);
  opacity: 0;
  animation: sg-drift linear infinite;
}}
.sg-ember:nth-child(1) {{ left: 6%;  animation-duration: 22s; animation-delay: 0s; }}
.sg-ember:nth-child(2) {{ left: 11%; animation-duration: 28s; animation-delay: -6s; width: 2px; height: 2px; }}
.sg-ember:nth-child(3) {{ left: 88%; animation-duration: 24s; animation-delay: -3s; }}
.sg-ember:nth-child(4) {{ left: 93%; animation-duration: 30s; animation-delay: -12s; width: 2px; height: 2px; }}
.sg-ember:nth-child(5) {{ left: 4%;  animation-duration: 26s; animation-delay: -9s; background: rgba(232,226,212,0.22); }}
.sg-ember:nth-child(6) {{ left: 96%; animation-duration: 32s; animation-delay: -15s; background: rgba(232,226,212,0.22); }}
@keyframes sg-drift {{
  0%   {{ transform: translate3d(0, -8vh, 0); opacity: 0; }}
  12%  {{ opacity: 0.45; }}
  80%  {{ opacity: 0.25; }}
  100% {{ transform: translate3d(12px, 108vh, 0); opacity: 0; }}
}}
@media (max-width: 1100px) {{
  .sg-embers {{ display: none; }}
}}
@media (max-width: 420px) {{
  p.sg-wordmark, .sg-wordmark {{ font-size: 28px !important; }}
  img.sg-logo, .sg-logo {{
    width: 52px !important;
    height: 52px !important;
    min-width: 52px !important;
    min-height: 52px !important;
    max-width: 52px !important;
    max-height: 52px !important;
  }}
}}
</style>
"""

WELCOME_CENTER_JS = f"""
<script>
(function () {{
  const RESERVE = {INPUT_BAR_RESERVE_PX};
  function apply() {{
    const marker = window.parent.document.querySelector('.sg-welcome-marker');
    const doc = window.parent.document;
    const containers = [
      ...doc.querySelectorAll('[data-testid="stMainBlockContainer"]'),
      ...doc.querySelectorAll('.block-container'),
    ];
    containers.forEach((el) => {{
      if (!marker || !el.contains(marker)) {{
        if (el.dataset.sgWelcome === '1') {{
          el.style.height = '';
          el.style.minHeight = '';
          el.style.maxHeight = '';
          el.style.display = '';
          el.style.flexDirection = '';
          el.style.justifyContent = '';
          el.style.paddingTop = '';
          el.style.overflow = '';
          delete el.dataset.sgWelcome;
          const child = el.firstElementChild;
          if (child && child.dataset && child.dataset.sgWelcomeChild === '1') {{
            child.style.display = '';
            child.style.flexDirection = '';
            child.style.justifyContent = '';
            child.style.flex = '';
            delete child.dataset.sgWelcomeChild;
          }}
        }}
        return;
      }}
      const bottom = doc.querySelector('[data-testid="stBottom"]');
      const bottomH = bottom ? Math.ceil(bottom.getBoundingClientRect().height) : RESERVE;
      const h = Math.max(240, window.parent.innerHeight - bottomH);
      el.dataset.sgWelcome = '1';
      el.style.boxSizing = 'border-box';
      el.style.height = h + 'px';
      el.style.minHeight = h + 'px';
      el.style.maxHeight = h + 'px';
      el.style.display = 'flex';
      el.style.flexDirection = 'column';
      el.style.justifyContent = 'center';
      el.style.paddingTop = '0';
      el.style.overflow = 'hidden';
      const child = el.firstElementChild;
      if (child) {{
        child.dataset.sgWelcomeChild = '1';
        child.style.display = 'flex';
        child.style.flexDirection = 'column';
        child.style.justifyContent = 'center';
        child.style.flex = '1 1 auto';
      }}
    }});
  }}
  apply();
  window.parent.addEventListener('resize', apply);
  new window.parent.MutationObserver(apply).observe(window.parent.document.body, {{
    childList: true, subtree: true,
  }});
}})();
</script>
"""

DETECT_ICON = (
    '<svg viewBox="0 0 24 24" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true">'
    '<circle cx="11" cy="11" r="7"/><path d="M21 21l-3.5-3.5"/></svg>'
)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


TAG_TO_BOSS = {_slug(n): n for n in BOSS_CLASSES}
TAG_TO_BOSS.update(
    {
        "corrupted_monk": "Corrupted Monk",
        "divine_dragon": "Divine Dragon",
        "genichiro": "Genichiro Phase 1",
        "guardian_ape": "Guardian Ape",
        "owl": "Owl",
    }
)


def _boss_short(name: str) -> str:
    if " - " in name:
        name = name.split(" - ")[-1]
    return name if len(name) <= 14 else name[:13] + "…"


def _boss_mono(name: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", name)
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[1][0]).upper()


def _init_state() -> None:
    if "conversations" not in st.session_state:
        cid = uuid.uuid4().hex[:8]
        st.session_state.conversations = {
            cid: {"id": cid, "title": "New conversation", "messages": [], "updated": 0}
        }
        st.session_state.active_id = cid
    if "conv_clock" not in st.session_state:
        st.session_state.conv_clock = 1
    # Migrate older single-thread sessions
    if "messages" in st.session_state and st.session_state.messages:
        active = st.session_state.conversations[st.session_state.active_id]
        if not active["messages"]:
            active["messages"] = list(st.session_state.messages)
            for msg in active["messages"]:
                if msg.get("role") == "user":
                    active["title"] = msg["content"].strip()[:56]
                    break
        del st.session_state.messages


def _active() -> dict:
    return st.session_state.conversations[st.session_state.active_id]


def _messages() -> list:
    return _active()["messages"]


def _new_conversation() -> None:
    cid = uuid.uuid4().hex[:8]
    st.session_state.conv_clock += 1
    st.session_state.conversations[cid] = {
        "id": cid,
        "title": "New conversation",
        "messages": [],
        "updated": st.session_state.conv_clock,
    }
    st.session_state.active_id = cid
    st.query_params.from_dict({"c": cid})


def _sync_query_params() -> None:
    params = st.query_params
    if params.get("new") == "1":
        _new_conversation()
        return
    cid = params.get("c")
    if cid and cid in st.session_state.conversations:
        st.session_state.active_id = cid


def _data_uri(path: Path, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _inject_chrome() -> None:
    parts = [CUSTOM_CSS]
    if BACKDROP_PATH.exists():
        uri = _data_uri(BACKDROP_PATH)
        embers = '<span class="sg-ember"></span>' * 6
        parts.append(
            f'<div class="sg-backdrop" style="background-image:url(\'{uri}\')"></div>'
            f'<div class="sg-scrim" aria-hidden="true"></div>'
            f'<div class="sg-embers" aria-hidden="true">{embers}</div>'
        )
    st.markdown("".join(parts), unsafe_allow_html=True)


def _health_payload() -> tuple[str, str, str]:
    """Return (dot_class, status_line, meta_line)."""
    try:
        status = api_client.health()
        store = status.get("vector_store", {})
        llm = status.get("llm", {})
        chunks = store.get("chunks", 0)
        model = llm.get("model") or "—"
        short_model = model.split("/")[-1] if isinstance(model, str) else str(model)
        meta = f"{short_model} · {chunks:,} chunks"
        if status.get("status") == "ok":
            return "ok", "Backend connected", meta
        return "warn", "Backend degraded", meta
    except api_client.ApiClientError:
        return "bad", "Backend unavailable", "—"


def _latest_detected_boss(messages: list) -> str | None:
    for msg in reversed(messages):
        det = msg.get("detection")
        if det is None:
            continue
        boss = getattr(det, "boss", None) or (det.get("boss") if isinstance(det, dict) else None)
        if not boss:
            continue
        return TAG_TO_BOSS.get(boss, TAG_TO_BOSS.get(_slug(boss), boss))
    return None


def _render_sidebar() -> None:
    """Custom fixed left rail (not Streamlit's sidebar).

    The rail is permanently open, so there is no open/closed state to keep:
    the markup is plain static HTML and the panel is positioned with CSS only.
    Nothing here can trigger a rerun or reset the chat.
    """
    dot, status_line, meta = _health_payload()
    hot = _latest_detected_boss(_messages())

    convs = sorted(
        st.session_state.conversations.values(),
        key=lambda c: c.get("updated", 0),
        reverse=True,
    )
    visible = [c for c in convs if c["messages"] or c["id"] == st.session_state.active_id]

    if not visible:
        hist_html = '<p class="sg-hist-empty">No conversations yet.</p>'
    else:
        links = []
        for conv in visible:
            title = conv["title"] or "New conversation"
            if len(title) > 42:
                title = title[:41] + "…"
            active = " is-active" if conv["id"] == st.session_state.active_id else ""
            links.append(
                f'<a class="sg-hist{active}" href="?c={escape(conv["id"])}" '
                f'target="_self" title="{escape(conv["title"])}">{escape(title)}</a>'
            )
        hist_html = "".join(links)

    bosses = []
    for name in BOSS_CLASSES:
        cls = (
            "sg-boss is-hot"
            if hot and (hot == name or _slug(hot) == _slug(name) or hot in name or name in hot)
            else "sg-boss"
        )
        bosses.append(
            f'<div class="{cls}" title="{escape(name)}">'
            f'<div class="sg-boss-orb">{escape(_boss_mono(name))}</div>'
            f'<span class="sg-boss-name">{escape(_boss_short(name))}</span>'
            f"</div>"
        )

    st.markdown(
        f"""
        <aside class="sg-drawer" id="sg-drawer">
          <p class="sg-side-kicker">Conversations</p>
          <a class="sg-new" href="?new=1" target="_self">+  New conversation</a>
          {hist_html}
          <div class="sg-side-gap"></div>
          <p class="sg-side-kicker">Boss roster</p>
          <div class="sg-roster">{"".join(bosses)}</div>
          <div class="sg-side-foot">
            <p class="sg-side-foot-line"><span class="sg-dot {dot}"></span>{escape(status_line)}</p>
            <p class="sg-side-foot-meta">{escape(meta)}</p>
          </div>
        </aside>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    logo = _data_uri(LOGO_PATH) if LOGO_PATH.exists() else ""
    logo_html = (
        f'<img class="sg-logo" src="{logo}" alt="Sekiro Geeker logo" '
        f'width="64" height="64" />'
        if logo
        else ""
    )
    st.markdown(
        f"""
        <div class="sg-header">
          <div class="sg-brand-row">
            {logo_html}
            <p class="sg-wordmark">Sekiro Geeker</p>
          </div>
          <p class="sg-tagline">
            Answers questions about Sekiro using a local wiki index, and can
            identify a boss from a screenshot to focus the answer on that fight.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_detection(detection) -> None:
    if detection is None:
        return
    focus = " · focused" if detection.used_for_retrieval else ""
    st.markdown(
        f'<div class="sg-detect">{DETECT_ICON}'
        f"Detected: <strong>{escape(detection.boss)}</strong> · "
        f"{detection.confidence:.0%}{focus}</div>",
        unsafe_allow_html=True,
    )


def _render_sources(sources: list[str]) -> None:
    if not sources:
        return
    items = "".join(f"<li><code>{escape(source)}</code></li>" for source in sources)
    st.markdown(
        f'<div class="sg-sources-wrap">'
        f'<details class="sg-sources">'
        f'<summary><span class="sg-chevron">▸</span> Sources ({len(sources)})</summary>'
        f"<ul>{items}</ul></details></div>",
        unsafe_allow_html=True,
    )


def _empty_prompts() -> None:
    st.markdown(
        '<p class="sg-empty-hint">Try one of these:</p>',
        unsafe_allow_html=True,
    )
    for question in EXAMPLE_QUESTIONS:
        if st.button(question, key=f"ex_{abs(hash(question))}"):
            st.session_state.pending = question


def _read_upload(files) -> tuple[bytes | None, str | None]:
    if not files:
        return None, None
    uploaded = files[0]
    return uploaded.getvalue(), uploaded.name


def main() -> None:
    _init_state()
    _sync_query_params()
    _inject_chrome()
    _render_sidebar()

    messages = _messages()
    empty = not messages
    if empty:
        st.markdown('<div class="sg-welcome-marker"></div>', unsafe_allow_html=True)

    _render_header()

    if empty:
        _empty_prompts()
        components.html(WELCOME_CENTER_JS, height=0, width=0)

    for message in messages:
        avatar = AVATAR_USER if message["role"] == "user" else AVATAR_ASSISTANT
        with st.chat_message(message["role"], avatar=avatar):
            if message["role"] == "assistant":
                _render_detection(message.get("detection"))
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_sources(message.get("sources", []))

    chat_value = st.chat_input(
        "Ask about Sekiro…",
        accept_file=True,
        file_type=["png", "jpg", "jpeg"],
        max_upload_size=10,
    )

    prompt = st.session_state.pop("pending", None)
    turn_image: bytes | None = None
    turn_name = "screenshot.png"

    if chat_value is not None:
        if isinstance(chat_value, str):
            prompt = chat_value
        else:
            prompt = (chat_value.text or "").strip() or prompt
            file_bytes, file_name = _read_upload(chat_value.files)
            if file_bytes is not None:
                turn_image = file_bytes
                turn_name = file_name or "screenshot.png"

    if not prompt:
        return

    conv = _active()
    conv["messages"].append({"role": "user", "content": prompt})
    if conv["title"] in ("", "New conversation"):
        conv["title"] = prompt.strip()[:56]
    st.session_state.conv_clock += 1
    conv["updated"] = st.session_state.conv_clock

    with st.chat_message("user", avatar=AVATAR_USER):
        st.markdown(prompt)
        if turn_image is not None:
            st.caption(f"Attached: {turn_name}")

    with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
        status = st.empty()
        status.markdown(
            '<p class="sg-searching">Searching the local wiki index…</p>',
            unsafe_allow_html=True,
        )
        try:
            result = api_client.ask(
                prompt,
                image_bytes=turn_image,
                image_name=turn_name,
            )
        except api_client.ApiClientError as exc:
            status.empty()
            st.error(str(exc))
            conv["messages"].pop()
            return

        status.empty()
        _render_detection(result.detection)
        st.markdown(result.answer)
        _render_sources(result.sources)

    conv["messages"].append(
        {
            "role": "assistant",
            "content": result.answer,
            "sources": result.sources,
            "detection": result.detection,
        }
    )
    st.rerun()


if __name__ == "__main__":
    main()
