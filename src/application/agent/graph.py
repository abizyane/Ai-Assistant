"""LangGraph agentic RAG: Retrieve → Generate → Verify.

Single-pass retrieval: the reranker filters chunks by relevance, so a
separate per-chunk grading loop is redundant and expensive.  The verify node
checks grounding and retries generation at most once.

Hard caps are enforced via state counters, never via recursion limits, so the
graph cannot loop forever even with adversarial verifiers. Node functions are
closures over injected use cases / LLM port, so the graph remains testable
with ``AsyncMock`` collaborators (no network in tests).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from langgraph.graph import END, StateGraph

from src.application.use_cases.generate_answer import GenerateAnswerUseCase
from src.application.use_cases.retrieve import RetrieveUseCase
from src.domain.entities.answer import AnswerCitation, AnswerWithCitations
from src.domain.ports.dto import GenerationRequest, RetrievedChunk
from src.domain.ports.llm import LLMPort
from src.shared.tracing import traced

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from src.config.settings import Settings

__all__ = ["AgentState", "build_agent_graph"]


class AgentState(TypedDict, total=False):
    """State carried through the agentic RAG graph.

    ``total=False`` lets each node return a partial dict that LangGraph merges
    into the running state.
    """

    query: str
    retrieved_chunks: list[RetrievedChunk]
    draft_answer: str | None
    citations: list[AnswerCitation]
    retry_count: int
    grounded: bool
    language: str
    session_id: str | None
    final_answer: AnswerWithCitations | None


def _make_retrieve_node(retrieve_uc: RetrieveUseCase) -> object:
    """Build the retrieve node bound to *retrieve_uc*."""

    @traced("agent.node.retrieve")
    async def retrieve_node(state: AgentState) -> dict[str, object]:
        chunks = await retrieve_uc.execute(
            query=state["query"],
            language=state.get("language"),
            session_id=state.get("session_id"),
        )
        return {"retrieved_chunks": chunks}

    return retrieve_node


def _make_generate_node(generate_uc: GenerateAnswerUseCase) -> object:
    """Build the answer-generation node bound to *generate_uc*."""

    @traced("agent.node.generate")
    async def generate_node(state: AgentState) -> dict[str, object]:
        chunks: list[RetrievedChunk] = state.get("retrieved_chunks", [])

        if not chunks:
            language = state.get("language", "en")
            no_ctx = AnswerWithCitations(
                text=(
                    "I couldn't find relevant information in the knowledge base"
                    " to answer your question."
                ),
                citations=[],
                language=language,
                tokens_in=0,
                tokens_out=0,
            )
            return {
                "draft_answer": no_ctx.text,
                "citations": [],
                "final_answer": no_ctx,
                "grounded": True,
            }

        answer = await generate_uc.execute(
            query=state["query"],
            retrieved_chunks=chunks,
            history=[],
            language=state.get("language", "en"),
        )
        return {
            "draft_answer": answer.text,
            "citations": list(answer.citations),
            "final_answer": answer,
            "grounded": False,
        }

    return generate_node


def _make_verify_node(grade_llm: LLMPort) -> object:
    """Build the grounding-verification node bound to *grade_llm*."""

    @traced("agent.node.verify_grounding")
    async def verify_grounding_node(state: AgentState) -> dict[str, object]:
        draft = state.get("draft_answer") or ""
        chunks: list[RetrievedChunk] = state.get("retrieved_chunks", [])

        if not draft or not chunks:
            return {"grounded": True}

        passages = "\n\n".join(f"- {c.content}" for c in chunks)
        prompt = (
            "You are a grounding judge. Decide whether the ANSWER is fully "
            "supported by the PASSAGES. Reply with a single token: 'yes' or "
            f"'no'.\n\nANSWER:\n{draft}\n\nPASSAGES:\n{passages}"
        )
        request = GenerationRequest(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="",
            temperature=0.0,
            max_tokens=4,
        )
        result = await grade_llm.generate(request)
        if result.text.strip().lower().startswith("y"):
            return {"grounded": True}
        return {
            "grounded": False,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    return verify_grounding_node


def _make_route_after_verify(max_retries: int) -> object:
    """Build the verify router bound to *max_retries*."""

    def _route(state: AgentState) -> str:
        if state.get("grounded", False):
            return END
        if state.get("retry_count", 0) > max_retries:
            return END
        return "generate"

    return _route


def build_agent_graph(
    retrieve_uc: RetrieveUseCase,
    generate_uc: GenerateAnswerUseCase,
    grade_llm: LLMPort,
    settings: Settings,
) -> CompiledStateGraph:
    """Build and compile the agentic RAG StateGraph.

    Wires retrieve \u2192 generate \u2192 verify_grounding into a compiled
    ``StateGraph``.  The returned graph is ready for ``ainvoke``/``astream``.

    Args:
        retrieve_uc: Injected retrieval use case.
        generate_uc: Injected answer-generation use case.
        grade_llm: LLM adapter used for grounding verification.
        settings: Application settings (used for loop bounds).

    Returns:
        Compiled ``CompiledStateGraph`` instance.
    """
    max_retries = settings.agent.max_regen_attempts

    sg: StateGraph = StateGraph(AgentState)
    sg.add_node("retrieve", _make_retrieve_node(retrieve_uc))
    sg.add_node("generate", _make_generate_node(generate_uc))
    sg.add_node("verify_grounding", _make_verify_node(grade_llm))

    sg.set_entry_point("retrieve")
    sg.add_edge("retrieve", "generate")
    sg.add_edge("generate", "verify_grounding")
    sg.add_conditional_edges(
        "verify_grounding",
        _make_route_after_verify(max_retries),
        {"generate": "generate", END: END},
    )

    return sg.compile()
