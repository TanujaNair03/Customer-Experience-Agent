"""Pytest coverage for the Agentic CX Support Router routing logic."""

from __future__ import annotations

from langchain_core.documents import Document

from agent import SupportRouterAgent


class FakeRetriever:
    """Deterministic retriever used to avoid external dependencies in tests."""

    def invoke(self, query: str) -> list[Document]:
        return [
            Document(
                page_content=(
                    "Standard shipping within the continental United States usually "
                    "takes 3 to 5 business days."
                )
            )
        ]


def test_standard_shipping_question_routes_to_rag_tool() -> None:
    agent = SupportRouterAgent(
        retriever=FakeRetriever(),
        classifier_fn=lambda _: "policy_question",
        answer_fn=lambda query, context: f"ANSWERED: {query} | {context}",
    )

    result = agent.invoke("How long does shipping take?")

    assert result["intent"] == "policy_question"
    assert result["route"] == "rag_tool"
    assert "3 to 5 business days" in result["retrieved_context"]
    assert "How long does shipping take?" in result["response_text"]


def test_angry_refund_demand_routes_to_human_escalation() -> None:
    agent = SupportRouterAgent(
        classifier_fn=lambda _: "angry_escalation",
    )

    result = agent.invoke("I hate this product, give me a refund right now!")

    assert result["intent"] == "angry_escalation"
    assert result["route"] == "human_escalation"
    assert result["escalation_payload"]["status"] == "escalated_to_human"
    assert result["escalation_payload"]["priority"] == "high"


def test_human_escalation_bypasses_retrieval() -> None:
    class FailingRetriever:
        def invoke(self, query: str) -> list[Document]:
            raise AssertionError("Retriever should not be called for escalations")

    agent = SupportRouterAgent(
        retriever=FailingRetriever(),
        classifier_fn=lambda _: "angry_escalation",
    )

    result = agent.invoke("This is ridiculous. Refund me immediately.")

    assert result["route"] == "human_escalation"
    assert result["escalation_payload"]["reason"] == "angry_or_refund_demand"


def test_escalation_payload_is_structured_json_serializable() -> None:
    agent = SupportRouterAgent(
        classifier_fn=lambda _: "angry_escalation",
        escalation_fn=lambda query: {
            "status": "escalated_to_human",
            "priority": "urgent",
            "reason": "custom_escalation_rule",
            "customer_message": query,
        },
    )

    result = agent.invoke("Your service is awful. I want a supervisor.")

    assert result["route"] == "human_escalation"
    assert result["response_text"].startswith("{")
    assert result["escalation_payload"]["priority"] == "urgent"
