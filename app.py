"""Streamlit UI for the Agentic CX Support Router demo."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from agent import build_default_agent, build_support_graph


st.set_page_config(
    page_title="Agentic CX Support Router",
    page_icon=":black_square_button:",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent
ASSISTANT_AVATAR = Path(
    BASE_DIR / "assets" / "chatbot_avatar.avif"
)
USER_AVATAR = BASE_DIR / "assets" / "user_avatar.svg"

STARTER_PROMPTS = [
    "What is your return policy?",
    "How long does shipping take?",
    "Can I exchange an item for another size?",
    "Do you ship internationally?",
    "My order arrived damaged. What should I do?",
    "I need to speak to a human agent.",
]

GREETING_TRIGGERS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}


def inject_styles() -> None:
    """Apply a minimal black-and-white design system to Streamlit."""
    st.markdown(
        """
        <style>
        .stApp {
            background: #f5f5f0;
            color: #111111;
        }
        [data-testid="stSidebar"] {
            background: #111111;
            border-right: 1px solid #d9d9d3;
        }
        [data-testid="stSidebar"] * {
            color: #f5f1e8 !important;
        }
        [data-testid="stSidebar"] .stButton > button {
            background: #f5f1e8;
            color: #111111 !important;
            border-color: #f5f1e8;
        }
        [data-testid="stSidebarNav"] {
            display: none;
        }
        [data-testid="stHeader"] {
            background: rgba(245, 245, 240, 0.92);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 8rem;
            max-width: 1120px;
        }
        .hero-wrap {
            background: #ffffff;
            border: 1px solid #111111;
            box-shadow: 10px 10px 0 #111111;
            padding: 1.7rem 1.8rem 1.45rem 1.8rem;
            margin-bottom: 1.4rem;
        }
        .hero-kicker {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #555555;
            margin-bottom: 0.6rem;
        }
        .hero-title {
            font-size: 3rem;
            line-height: 1;
            font-weight: 700;
            letter-spacing: -0.04em;
            margin-bottom: 0.75rem;
        }
        .hero-copy {
            font-size: 1rem;
            line-height: 1.6;
            max-width: 760px;
            color: #2e2e2e;
        }
        .hint-card {
            background: #111111;
            color: #ffffff;
            padding: 1rem 1.1rem;
            margin: 0 0 1rem 0;
        }
        .hint-card strong {
            display: block;
            margin-bottom: 0.35rem;
        }
        .meta-card {
            border: 1px solid #111111;
            background: #ffffff;
            padding: 0.95rem 1rem;
            margin-top: 0.65rem;
        }
        .meta-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.6rem 1rem;
        }
        .meta-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #666666;
        }
        .meta-value {
            font-size: 0.98rem;
            color: #111111;
            line-height: 1.45;
        }
        .faq-answer {
            color: #111111 !important;
        }
        [data-testid="stChatMessage"] {
            background: transparent;
            padding-left: 0;
            padding-right: 0;
        }
        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"] {
            width: 4.25rem !important;
            height: 4.25rem !important;
            min-width: 4.25rem !important;
            min-height: 4.25rem !important;
            border-radius: 0 !important;
            overflow: hidden !important;
            box-shadow: none !important;
            background: transparent !important;
        }
        [data-testid="stChatMessageAvatarUser"] *,
        [data-testid="stChatMessageAvatarAssistant"] * {
            background: transparent !important;
            box-shadow: none !important;
            border-radius: 0 !important;
        }
        [data-testid="stChatMessageAvatarUser"] img,
        [data-testid="stChatMessageAvatarAssistant"] img {
            width: 100% !important;
            height: 100% !important;
            object-fit: contain !important;
            background: transparent !important;
            border-radius: 0 !important;
        }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
            font-size: 1.04rem;
            line-height: 1.75;
            color: #111111;
        }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ul,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
            color: #111111;
            font-size: 1rem;
            line-height: 1.75;
        }
        [data-testid="stBottomBlockContainer"] {
            background: #111111;
            border-top: 1px solid #111111;
            padding-top: 1rem;
            padding-bottom: 1.2rem;
        }
        [data-testid="stBottomBlockContainer"] > div {
            background: transparent !important;
        }
        [data-testid="stChatInput"] {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
        }
        [data-testid="stChatInput"] > div {
            background: #f5f1e8 !important;
            border: 1px solid #111111 !important;
            border-radius: 999px !important;
            box-shadow: none !important;
            padding-left: 0.35rem;
            padding-right: 0.35rem;
        }
        [data-testid="stChatInput"] input {
            background: transparent !important;
            color: #111111 !important;
        }
        [data-testid="stChatInput"] button {
            background: #111111 !important;
            color: #f5f1e8 !important;
            border-radius: 999px !important;
        }
        .stButton > button {
            width: 100%;
            border-radius: 999px;
            border: 1px solid #111111;
            background: #ffffff;
            color: #111111;
            padding: 0.55rem 0.9rem;
        }
        .stButton > button:hover {
            background: #111111;
            color: #ffffff;
            border-color: #111111;
        }
        .top-actions {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 0.7rem;
        }
        .chat-shell {
            margin-top: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def assistant_avatar() -> str:
    """Return the assistant avatar path, with a safe fallback."""
    return str(ASSISTANT_AVATAR) if ASSISTANT_AVATAR.exists() else "🤖"


def user_avatar() -> str:
    """Return the user avatar path, with a safe fallback."""
    return str(USER_AVATAR) if USER_AVATAR.exists() else "🙂"


def build_welcome_message() -> str:
    """Return the opening assistant message."""
    return (
        "Hi, how can I help you today?\n\n"
        "I can help with return policy and refunds, shipping times, exchanges, "
        "international shipping, damaged or incorrect items, and connecting you "
        "to a human support specialist."
    )


def initialize_session_state() -> None:
    """Create session defaults for chat history and reusable agent instance."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": build_welcome_message(),
                "route": "welcome",
            }
        ]
    if "agent" not in st.session_state:
        st.session_state.agent = build_default_agent()
    if "support_graph" not in st.session_state:
        st.session_state.support_graph = build_support_graph()
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


def is_greeting(message: str) -> bool:
    """Return True when the user message is a simple greeting."""
    return message.strip().lower() in GREETING_TRIGGERS


def render_sidebar() -> None:
    """Explain the architecture in the sidebar."""
    st.sidebar.markdown("## About")
    st.sidebar.markdown(
        """
This app demonstrates an agentic CX workflow:

- LangGraph routes each customer message
- Gemini classifies intent and drafts answers
- ChromaDB retrieves grounded FAQ context
- urgent or angry cases are handed to a human queue
"""
    )
    st.sidebar.markdown("## Try Asking")
    st.sidebar.markdown(
        """
- Ask about returns, shipping, exchanges, or damaged items
- Try an angry refund demand to test escalation
- Click a starter action instead of typing
"""
    )
    st.sidebar.markdown("## Human Handoff")
    st.sidebar.markdown(
        """
When escalation is required, the app creates a handoff payload with:

- ticket ID
- priority
- queue
- customer message
- recommended next step

In a production system, that payload would be sent to Zendesk, Salesforce, or an internal support queue.
"""
    )


def render_starter_actions() -> None:
    """Show clickable quick actions for common support journeys."""
    st.markdown(
        """
        <div class="hint-card">
            <strong>Choose a quick path</strong>
            Start with a common CX question or jump straight to a human handoff.
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(3, gap="small")
    for index, prompt in enumerate(STARTER_PROMPTS):
        if columns[index % 3].button(prompt, key=f"starter_{index}", use_container_width=True):
            st.session_state.pending_prompt = prompt


def render_message_metadata(message: dict[str, object]) -> None:
    """Render handoff metadata when needed."""
    escalation_payload = message.get("escalation_payload")
    if isinstance(escalation_payload, dict):
        st.markdown(
            f"""
            <div class="meta-card">
                <div class="meta-grid">
                    <div>
                        <div class="meta-label">Ticket ID</div>
                        <div class="meta-value">{escalation_payload.get("ticket_id", "Pending")}</div>
                    </div>
                    <div>
                        <div class="meta-label">Priority</div>
                        <div class="meta-value">{escalation_payload.get("priority", "high")}</div>
                    </div>
                    <div>
                        <div class="meta-label">Queue</div>
                        <div class="meta-value">{escalation_payload.get("queue", "cx_priority_support")}</div>
                    </div>
                    <div>
                        <div class="meta-label">Estimated Response Time</div>
                        <div class="meta-value">{escalation_payload.get("estimated_response_time", "within 15 minutes")}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_chat_history() -> None:
    """Draw the chat transcript from Streamlit session state."""
    for message in st.session_state.messages:
        avatar = assistant_avatar() if message["role"] == "assistant" else user_avatar()
        with st.chat_message(message["role"], avatar=avatar):
            render_message_metadata(message)
            st.markdown(message["content"])


def run_agent(user_prompt: str) -> dict[str, object]:
    """Invoke the router and return the full agent result."""
    _ = st.session_state.support_graph
    try:
        return st.session_state.agent.invoke(user_prompt)
    except Exception:
        return {
            "response_text": (
                "I’m sorry, something went wrong while handling your request. "
                "Please try again in a moment."
            ),
            "error": "unexpected_error",
        }


def process_user_prompt(user_prompt: str) -> None:
    """Append the user message, run the workflow, and store the assistant response."""
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    if is_greeting(user_prompt):
        assistant_result = {
            "response_text": build_welcome_message(),
            "route": "welcome",
        }
    else:
        assistant_result = run_agent(user_prompt)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_result["response_text"],
            "route": assistant_result.get("route"),
            "escalation_payload": assistant_result.get("escalation_payload"),
        }
    )


def main() -> None:
    """Render the Streamlit app."""
    inject_styles()
    initialize_session_state()
    render_sidebar()

    pending_prompt = st.session_state.pending_prompt
    if pending_prompt:
        st.session_state.pending_prompt = None
        process_user_prompt(pending_prompt)

    user_prompt = st.chat_input("Ask about returns, shipping, exchanges, or a refund issue...")
    if user_prompt:
        process_user_prompt(user_prompt)
        st.rerun()

    top_actions = st.columns([1, 0.22], gap="small")
    with top_actions[1]:
        if st.button("Reset Chat", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": build_welcome_message(), "route": "welcome"}
            ]
            st.session_state.pending_prompt = None
            st.rerun()

    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-kicker">Agentic CX Workflow Demo</div>
            <div class="hero-title">Agentic CX Support Router</div>
            <div class="hero-copy">
                A minimal customer support experience built with LangGraph, Gemini, and ChromaDB.
                The app guides the user, answers grounded FAQ questions, and escalates sensitive cases
                to a human support flow with visible handoff metadata.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_starter_actions()
    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    render_chat_history()
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
