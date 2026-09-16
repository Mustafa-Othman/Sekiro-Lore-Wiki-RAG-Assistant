"""Capture README screenshots of the running app with a headless browser.

Assumes the backend (:8000) and Streamlit (:8501) are already up. Writes PNGs to
docs/screenshots/. This is a documentation tool, not part of the app -- it is not
in either requirements.txt.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:8501"
QUESTION = "Who is the Sculptor, and what is his backstory?"


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(URL, wait_until="networkidle", timeout=90_000)

        # Streamlit hydrates over websocket; wait for the chat input to exist.
        page.wait_for_selector('[data-testid="stChatInput"] textarea', timeout=90_000)
        page.wait_for_timeout(4_000)

        # Empty state: the left rail's status footer + welcome copy.
        page.screenshot(path=str(OUT / "01-empty-state.png"), full_page=True)
        print("captured 01-empty-state.png")

        # Ask the obscure question that the grounded pipeline answers correctly.
        box = page.locator('[data-testid="stChatInput"] textarea')
        box.click()
        box.fill(QUESTION)
        box.press("Enter")
        print(f"asked: {QUESTION}")

        # While the answer is in flight the assistant message shows the
        # "searching" marker. Grab it here: this is the in-between state, and
        # the avatar has to stay at full opacity in it.
        try:
            page.wait_for_selector(".sg-searching", state="visible", timeout=120_000)
            page.wait_for_timeout(400)
            page.screenshot(path=str(OUT / "05-thinking.png"), full_page=True)
            print("captured 05-thinking.png")
        except Exception as exc:
            print(f"thinking shot skipped: {exc.__class__.__name__}")

        # The answer arrives with a "Sources (n)" expander. A local 7B on CPU is
        # slow, so allow generously; the spinner shows meanwhile.
        try:
            page.wait_for_selector("text=/Sources \\(/", timeout=240_000)
        except Exception as exc:
            print(f"WARNING: sources expander never appeared ({exc.__class__.__name__})")

        page.wait_for_timeout(3_000)
        page.screenshot(path=str(OUT / "02-grounded-answer.png"), full_page=True)
        print("captured 02-grounded-answer.png")

        # Expand the citations, so the screenshot shows the grounding evidence.
        try:
            page.locator("text=/Sources \\(/").first.click()
            page.wait_for_timeout(1_500)
            page.screenshot(path=str(OUT / "03-cited-sources.png"), full_page=True)
            print("captured 03-cited-sources.png")
        except Exception as exc:
            print(f"could not expand sources: {exc.__class__.__name__}")

        # Scroll the chat into view for a tighter shot of the answer itself.
        try:
            page.locator('[data-testid="stChatMessage"]').last.scroll_into_view_if_needed()
            page.wait_for_timeout(1_000)
            page.screenshot(path=str(OUT / "04-answer-closeup.png"))
            print("captured 04-answer-closeup.png")
        except Exception as exc:
            print(f"closeup skipped: {exc.__class__.__name__}")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
