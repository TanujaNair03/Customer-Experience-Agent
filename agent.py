"""LangGraph orchestration for an Agentic CX Support Router demo."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Protocol
from uuid import uuid4

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from typing_extensions import NotRequired, TypedDict

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
DEFAULT_CHAT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

FAQ_FALLBACKS = {
    "return": (
        "You can return unworn apparel and footwear within 30 days of delivery. "
        "Items must include the original tags and packaging. Refunds are issued "
        "to the original payment method within 5 to 7 business days after the "
        "return is inspected. Final sale items and personalized products cannot "
        "be returned unless they arrive damaged or defective."
    ),
    "shipping": (
        "Standard shipping within the continental United States usually takes 3 to "
        "5 business days. Expedited shipping usually takes 1 to 2 business days. "
        "Orders placed after 3 PM Eastern Time are processed on the next business day."
    ),
    "exchange": (
        "We currently do not offer direct exchanges. Please return the original "
        "item and place a new order for the replacement size or color."
    ),
    "international": (
        "International shipping is available to select countries and usually takes "
        "7 to 14 business days. Customs delays may extend the final delivery time. "
        "A human agent would need to verify the exact country list."
    ),
    "damaged": (
        "If a package arrives damaged or contains the wrong item, please contact "
        "support within 7 days of delivery. The support team will provide a prepaid "
        "return label and prioritize a replacement or refund review."
    ),
}

COMPLIMENT_KEYWORDS = {
    "love",
    "loved",
    "amazing",
    "great",
    "wonderful",
    "awesome",
    "excellent",
    "thanks",
    "thank you",
    "nice",
    "pretty",
    "beautiful",
    "lovely",
    "cute",
}

COUNTRY_HINTS = {
    "india",
    "argentina",
    "canada",
    "australia",
    "uk",
    "united kingdom",
    "germany",
    "france",
    "singapore",
    "uae",
    "dubai",
    "mexico",
    "brazil",
    "japan",
}

ANGRY_KEYWORDS = {
    "terrible",
    "awful",
    "hate",
    "refund now",
    "ridiculous",
    "worst",
    "frustrating",
    "bad service",
    "angry",
    "complaint",
    "supervisor",
    "human agent",
    "horrible",
    "horrid",
    "never gonna recommend",
    "never recommend",
    "not recommend",
    "never coming back",
    "not coming back",
    "wont come back",
    "won't come back",
    "useless",
    "pathetic",
    "disappointed",
    "disappointing",
}

NEGATIVE_PATTERNS = [
    r"\bhorr\w*\b",
    r"\bawful\b",
    r"\bterrible\b",
    r"\bbad\b",
    r"\bworse\b",
    r"\bworst\b",
    r"\bpathetic\b",
    r"\buseless\b",
    r"\btrash\b",
    r"\bgarbage\b",
    r"\bdisappoint\w*\b",
    r"\bfrustrat\w*\b",
    r"\bcomplain\w*\b",
    r"\bangr\w*\b",
    r"\brefund\b",
    r"\bsupervisor\b",
    r"\bhuman agent\b",
    r"\bnever\s+(coming back|again|recommend)\b",
    r"\bnot\s+recommend\b",
    r"\bnot\s+coming\s+back\b",
    r"\bwon[' ]?t\s+come\s+back\b",
    r"\brefund me\b",
    r"\bfurious\b",
    r"\bdisgust\w*\b",
    r"\bappall\w*\b",
    r"\bpoor\b.*\b(service|experience)\b",
]

POSITIVE_PATTERNS = [
    r"\blov\w*\b",
    r"\bamazing\b",
    r"\bawesome\b",
    r"\bgreat\b",
    r"\bexcellent\b",
    r"\bwonderful\b",
    r"\bpretty\b",
    r"\bbeautiful\b",
    r"\blovely\b",
    r"\bcute\b",
    r"\bthank\w*\b",
    r"\bnice\b",
    r"\bfantastic\b",
    r"\bstunning\b",
    r"\bgorgeous\b",
]

Intent = Literal["policy_question", "angry_escalation"]
Route = Literal["rag_tool", "human_escalation"]


class AgentState(TypedDict):
    """State shared across the LangGraph workflow."""

    user_query: str
    intent: NotRequired[Intent]
    route: NotRequired[Route]
    retrieved_context: NotRequired[str]
    response_text: NotRequired[str]
    escalation_payload: NotRequired[dict[str, Any]]
    error: NotRequired[str]


class RetrieverLike(Protocol):
    """Minimal retriever protocol used by the agent for testability."""

    def invoke(self, query: str) -> list[Document]:
        """Return relevant documents for a query."""


ClassifierFn = Callable[[str], Intent]
AnswerFn = Callable[[str, str], str]
EscalationFn = Callable[[str], dict[str, Any]]


class APIRateLimitError(RuntimeError):
    """Raised when Gemini rejects a request due to rate limiting."""


class KnowledgeBaseError(RuntimeError):
    """Raised when the FAQ vector store or embedding step is unavailable."""


def require_google_api_key() -> str:
    """Load the Gemini API key or raise a helpful error."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Export your Google Gemini API key before "
            "running the support router."
        )
    return api_key


def parse_json_response(raw_text: str) -> dict[str, Any]:
    """Parse JSON content from a model response with a helpful failure mode."""
    cleaned_text = raw_text.strip()
    if cleaned_text.startswith("```"):
        lines = cleaned_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned_text = "\n".join(lines).strip()

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Expected valid JSON from Gemini, received: {raw_text}") from exc


def extract_intent_label(raw_text: str) -> Intent:
    """Extract an intent label from plain text or fenced JSON-like model output."""
    cleaned_text = raw_text.strip()
    if cleaned_text.startswith("```"):
        lines = cleaned_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned_text = "\n".join(lines).strip()

    try:
        payload = json.loads(cleaned_text)
        if isinstance(payload, dict):
            intent = payload.get("intent")
            if intent in {"policy_question", "angry_escalation"}:
                return intent
    except json.JSONDecodeError:
        pass

    match = re.search(r"(policy_question|angry_escalation)", cleaned_text)
    if match:
        return match.group(1)  # type: ignore[return-value]

    raise ValueError(f"Could not extract intent label from Gemini response: {raw_text}")


def is_rate_limit_error(exc: Exception) -> bool:
    """Detect common Gemini rate limit errors without depending on one SDK type."""
    message = str(exc).lower()
    return (
        "429" in message
        or "resource_exhausted" in message
        or "rate limit" in message
        or "quota" in message
    )


def heuristic_intent(query: str) -> Intent:
    """Classify obvious escalations without needing an LLM call."""
    text = query.lower()
    if any(keyword in text for keyword in ANGRY_KEYWORDS):
        return "angry_escalation"
    if any(re.search(pattern, text) for pattern in NEGATIVE_PATTERNS):
        return "angry_escalation"
    return "policy_question"


def is_compliment(query: str) -> bool:
    """Detect positive sentiment more broadly than exact keyword matches."""
    text = query.lower()
    return any(keyword in text for keyword in COMPLIMENT_KEYWORDS) or any(
        re.search(pattern, text) for pattern in POSITIVE_PATTERNS
    )


def heuristic_answer(query: str) -> str:
    """Return a deterministic FAQ-style answer for common support topics."""
    text = query.lower()
    mentions_country = (
        any(country in text for country in COUNTRY_HINTS)
        or bool(re.search(r"\bin\s+[A-Z][a-z]+", query))
        or "international" in text
        or "country" in text
        or "abroad" in text
    )
    if is_compliment(query):
        if mentions_country or "ship" in text or "deliver" in text or "get them" in text:
            return FAQ_FALLBACKS["international"]
        return "Thanks so much. Do let me know how I can help you today."
    if "return" in text or "refund" in text:
        return FAQ_FALLBACKS["return"]
    if (
        "shipping" in text
        or "delivery" in text
        or "ship" in text
        or "deliver" in text
        or "get them" in text
        or mentions_country
    ):
        if mentions_country:
            return FAQ_FALLBACKS["international"]
        return FAQ_FALLBACKS["shipping"]
    if "exchange" in text or "size" in text or "color" in text:
        return FAQ_FALLBACKS["exchange"]
    if "damaged" in text or "wrong item" in text or "incorrect" in text:
        return FAQ_FALLBACKS["damaged"]
    if mentions_country:
        return FAQ_FALLBACKS["international"]
    return (
        "I can currently help with returns, shipping, exchanges, international delivery, "
        "damaged items, and human support handoff. Please ask about one of those topics."
    )


def heuristic_context(query: str) -> str:
    """Return a simple source snippet used when retrieval is unavailable."""
    answer = heuristic_answer(query)
    return f"Fallback FAQ context:\n{answer}"


class SupportRouterAgent:
    """Encapsulates the LangGraph routing workflow and external dependencies."""

    def __init__(
        self,
        retriever: RetrieverLike | None = None,
        classifier_fn: ClassifierFn | None = None,
        answer_fn: AnswerFn | None = None,
        escalation_fn: EscalationFn | None = None,
        chat_model: str = DEFAULT_CHAT_MODEL,
        persist_directory: Path = CHROMA_DIR,
    ) -> None:
        self.chat_model = chat_model
        self.persist_directory = persist_directory
        self._retriever = retriever
        self._classifier_fn = classifier_fn
        self._answer_fn = answer_fn
        self._escalation_fn = escalation_fn
        self.graph = self._build_graph()

    def _default_llm(self) -> ChatGoogleGenerativeAI:
        require_google_api_key()
        return ChatGoogleGenerativeAI(
            model=self.chat_model,
            google_api_key=os.environ["GOOGLE_API_KEY"],
            temperature=0,
            retries=0,
            request_timeout=10,
        )

    def _default_retriever(self) -> RetrieverLike:
        require_google_api_key()
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=os.environ["GOOGLE_API_KEY"],
        )
        vector_store = Chroma(
            persist_directory=str(self.persist_directory),
            collection_name="sportswear_faq",
            embedding_function=embeddings,
        )
        return vector_store.as_retriever(search_kwargs={"k": 3})

    def classify_intent(self, query: str) -> Intent:
        """Classify the query as FAQ-friendly or requiring human escalation."""
        if self._classifier_fn is not None:
            return self._classifier_fn(query)

        llm = self._default_llm()
        prompt = f"""
You are an intent classifier for a customer support router.
Classify the user's message into exactly one label:
- "policy_question" for normal questions about policies, shipping, exchanges, or order issues
- "angry_escalation" for angry, abusive, highly emotional, urgent refund demands, or cases that should go directly to a human

Return valid JSON only with this schema:
{{"intent": "policy_question" | "angry_escalation"}}

User message:
{query}
""".strip()
        try:
            result = llm.invoke(prompt)
        except Exception as exc:
            if is_rate_limit_error(exc):
                return heuristic_intent(query)
            raise
        raw_content = result.content if hasattr(result, "content") else str(result)
        return extract_intent_label(raw_content)

    def generate_policy_answer(self, query: str, context: str) -> str:
        """Generate a concise CX answer grounded in retrieved FAQ context."""
        if self._answer_fn is not None:
            return self._answer_fn(query, context)

        llm = self._default_llm()
        prompt = f"""
You are a helpful customer support assistant for SwiftStride Sportswear.
Use only the provided FAQ context to answer the customer.
If the answer is not present in the context, clearly say that this assistant can help with returns, shipping, exchanges, damaged items, and international delivery only, then offer a human handoff.
If the customer is sharing praise or casual feedback rather than asking for support, thank them briefly and remind them what topics you can help with.
Keep the tone polite, concise, and professional.

FAQ context:
{context}

Customer question:
{query}
""".strip()
        try:
            result = llm.invoke(prompt)
        except Exception as exc:
            if is_rate_limit_error(exc):
                return heuristic_answer(query)
            raise
        return result.content.strip()

    def build_escalation_payload(self, query: str) -> dict[str, Any]:
        """Create the structured response sent to a human support queue."""
        if self._escalation_fn is not None:
            return self._escalation_fn(query)

        return {
            "ticket_id": f"CX-{uuid4().hex[:8].upper()}",
            "status": "escalated_to_human",
            "priority": "high",
            "queue": "cx_priority_support",
            "reason": "angry_or_refund_demand",
            "customer_message": query,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "handoff_summary": (
                "Customer used frustrated or urgent language and should be handled by "
                "a human CX specialist."
            ),
            "recommended_next_step": "Assign to a human CX specialist for immediate follow-up.",
            "estimated_response_time": "within 15 minutes",
        }

    def _route_after_intent(self, state: AgentState) -> str:
        if state["intent"] == "policy_question":
            return "rag_tool"
        return "human_escalation"

    def _intent_router_node(self, state: AgentState) -> AgentState:
        try:
            intent = self.classify_intent(state["user_query"])
        except Exception as exc:
            if is_rate_limit_error(exc):
                raise APIRateLimitError(
                    "API rate limit reached, please wait a moment."
                ) from exc
            raise
        return {"intent": intent}

    def _rag_tool_node(self, state: AgentState) -> AgentState:
        try:
            retriever = self._retriever or self._default_retriever()
            documents = retriever.invoke(state["user_query"])
            context = "\n\n".join(doc.page_content for doc in documents)
            answer = self.generate_policy_answer(state["user_query"], context)
        except APIRateLimitError:
            raise
        except Exception as exc:
            context = heuristic_context(state["user_query"])
            answer = heuristic_answer(state["user_query"])
        return {
            "route": "rag_tool",
            "retrieved_context": context,
            "response_text": answer,
        }

    def _human_escalation_node(self, state: AgentState) -> AgentState:
        payload = self.build_escalation_payload(state["user_query"])
        return {
            "route": "human_escalation",
            "escalation_payload": payload,
            "response_text": (
                "I’m sorry this has been frustrating. Please give us a moment while "
                "we connect you to a human support specialist."
            ),
        }

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("intent_router", self._intent_router_node)
        workflow.add_node("rag_tool", self._rag_tool_node)
        workflow.add_node("human_escalation", self._human_escalation_node)

        workflow.set_entry_point("intent_router")
        workflow.add_conditional_edges(
            "intent_router",
            self._route_after_intent,
            {
                "rag_tool": "rag_tool",
                "human_escalation": "human_escalation",
            },
        )
        workflow.add_edge("rag_tool", END)
        workflow.add_edge("human_escalation", END)
        return workflow.compile()

    def invoke(self, user_query: str) -> AgentState:
        """Run the graph for a single user query."""
        try:
            return self.graph.invoke({"user_query": user_query})
        except APIRateLimitError as exc:
            return {
                "user_query": user_query,
                "response_text": str(exc),
                "error": "rate_limit",
            }
        except KnowledgeBaseError as exc:
            return {
                "user_query": user_query,
                "response_text": str(exc),
                "error": "knowledge_base_unavailable",
            }
        except Exception:
            return {
                "user_query": user_query,
                "response_text": (
                    "I’m sorry, something went wrong while routing your request. "
                    "Please try again in a moment."
                ),
                "error": "unexpected_error",
            }


def build_default_agent() -> SupportRouterAgent:
    """Factory used by scripts or demos that want the full Gemini + Chroma stack."""
    return SupportRouterAgent()


def build_support_graph():
    """Expose a compiled graph for UI integrations such as Streamlit."""
    return build_default_agent().graph


def main() -> None:
    """Simple CLI entrypoint for manual testing."""
    agent = build_default_agent()
    query = input("Customer message: ").strip()
    result = agent.invoke(query)
    print(result["response_text"])


if __name__ == "__main__":
    main()
