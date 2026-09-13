"""Streamlit chat UI for the Sekiro wiki RAG assistant.

Run with:

    cd frontend && streamlit run app.py

The backend URL comes from `API_BASE_URL` (see `.env.example`) and is never
hard-coded here -- `api_client` owns that lookup.

Core Track: type a question, get a grounded answer plus the wiki pages it came
from. Extended Track: attach a screenshot and retrieval is biased toward the boss
in it, when detection is enabled on the backend.
"""

from __future__ import annotations

import streamlit as st

import api_client

st.set_page_config(
    page_title="Sekiro Lore Assistant",
    page_icon="🗡️",
    layout="centered",
)

EXAMPLE_QUESTIONS = [
    "What is the Mortal Blade and what does it do?",
    "Who is the Sculptor, and what is his backstory?",
    "What are the requirements for the Purification ending?",
    "What are the phases of the Guardian Ape fight?",
]


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _render_sources(sources: list[str]) -> None:
    """Citation list -- the evidence the answer is grounded in."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            st.markdown(f"- {source}")


def _render_detection(detection) -> None:
    """Show what the detector saw, and whether it steered retrieval."""
    if detection is None:
        return
    if detection.used_for_retrieval:
        st.caption(
            f"🎯 Detected **{detection.boss}** ({detection.confidence:.0%}) "
            f"— retrieval focused on this boss."
        )
    else:
        st.caption(
            f"🎯 Detected **{detection.boss}** ({detection.confidence:.0%}) "
            f"— below the confidence threshold, so a normal search was used."
        )


def _sidebar() -> bytes | None:
    """Connection status, screenshot upload, and example questions."""
    with st.sidebar:
        st.header("Connection")
        # Defaults, so a failed health check below still leaves the rest of the
        # sidebar working instead of raising NameError.
        status: dict = {}
        detection: dict = {}
        try:
            status = api_client.health()
            if status.get("status") == "ok":
                st.success("Backend ready")
            else:
                st.warning("Backend degraded")

            store = status.get("vector_store", {})
            st.caption(
                f"Index: {store.get('chunks', 0):,} chunks · "
                f"{len(store.get('boss_classes', []))} boss classes"
            )

            llm = status.get("llm", {})
            if llm.get("available"):
                st.caption(f"LLM: `{llm.get('model')}`")
            else:
                st.caption("⚠️ LLM unavailable — answers disabled")

            detection = status.get("detection", {})
            st.caption(
                "Boss detection: enabled" if detection.get("enabled")
                else "Boss detection: off (text-only mode)"
            )
        except api_client.ApiClientError as exc:
            st.error(str(exc))
            st.caption(f"Expected backend at `{api_client.get_base_url()}`")

        st.divider()

        st.subheader("Screenshot (optional)")
        uploaded = st.file_uploader(
            "Attach a boss screenshot to focus retrieval",
            type=["png", "jpg", "jpeg"],
            help="Requires the Extended Track detector. Without it, the question "
                 "is answered by normal search.",
        )
        image_bytes = uploaded.getvalue() if uploaded is not None else None
        if uploaded is not None and not detection.get("enabled", False):
            st.caption("Detector is off — this image will be ignored.")

        st.divider()

        st.subheader("Try asking")
        for question in EXAMPLE_QUESTIONS:
            if st.button(question, width="stretch"):
                st.session_state.pending = question

    return image_bytes


def main() -> None:
    _init_state()
    image_bytes = _sidebar()

    st.title("🗡️ Sekiro Lore Assistant")
    st.caption(
        "Answers come only from the indexed Sekiro wiki. Every response cites the "
        "pages it used."
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            _render_detection(message.get("detection"))
            _render_sources(message.get("sources", []))

    # A sidebar example button, or the chat box itself.
    question = st.session_state.pop("pending", None) or st.chat_input("Ask about Sekiro…")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Deliberately explicit that this is a local index lookup, not a web
        # search -- "searching the wiki" reads like an internet request.
        with st.spinner("Searching the local wiki index…"):
            try:
                result = api_client.ask(question, image_bytes=image_bytes)
            except api_client.ApiClientError as exc:
                st.error(str(exc))
                # Not appended to history: a failure is not a turn worth keeping.
                st.session_state.messages.pop()
                return

        st.markdown(result.answer)
        _render_detection(result.detection)
        _render_sources(result.sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "sources": result.sources,
            "detection": result.detection,
        }
    )


if __name__ == "__main__":
    main()
