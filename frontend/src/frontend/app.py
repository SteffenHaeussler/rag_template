import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Chat", layout="centered")
st.title("RAG Chat")
st.caption("Ask questions about the ingested documents")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query backend
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"question": prompt, "top_k": 3},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                answer = data["answer"]
                sources = data["sources"]

                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for src in sources:
                            st.markdown(
                                f"**{src['filename']}** (score: {src['score']:.2f})"
                            )
                            st.text(src["text"][:300] + ("..." if len(src["text"]) > 300 else ""))
                            st.divider()

                # Build assistant message for history
                assistant_msg = answer
                if sources:
                    source_list = ", ".join(s["filename"] for s in sources)
                    assistant_msg += f"\n\n*Sources: {source_list}*"

                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_msg}
                )

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend. Is it running?")
            except requests.exceptions.HTTPError as e:
                st.error(f"Backend error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        "This is a RAG (Retrieval-Augmented Generation) chat interface. "
        "It searches through ingested documents and uses AI to generate answers."
    )

    # Health check
    if st.button("Check Backend Status"):
        try:
            resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
            data = resp.json()
            if data["qdrant"] == "connected":
                st.success(f"Backend: {data['status']}, Qdrant: {data['qdrant']}")
            else:
                st.warning(f"Backend: {data['status']}, Qdrant: {data['qdrant']}")
        except Exception:
            st.error("Backend is not reachable.")
