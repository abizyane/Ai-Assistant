"""LangGraph agentic RAG: Rewrite -> Retrieve -> Grade -> Generate -> Verify.

Bounded agentic loop:

* At most one query rewrite (``rewrite_count <= 1``).
* At most one regeneration retry on ungrounded drafts (``retry_count <= 1``).

Hard caps are enforced via state counters, never via recursion limits, so the
graph cannot loop forever even with adversarial graders. Node functions are
closures over injected use cases / LLM port, so the graph remains testable
with ``AsyncMock`` collaborators (no network in tests).
"""

from __future__ import annotations

import asyncio
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
    rewritten_query: str | None
    retrieved_chunks: list[RetrievedChunk]
    relevance_grades: list[bool]
    draft_answer: str | None
    citations: list[AnswerCitation]
    retry_count: int
    rewrite_count: int
    grounded: bool
    language: str
    session_id: str | None
    final_answer: AnswerWithCitations | None


def _make_rewrite_node(grade_llm: LLMPort) -> object:
    """Build the rewrite-query node bound to *grade_llm*."""

    @traced("agent.node.rewrite_query")
    async def rewrite_query_node(state: AgentState) -> dict[str, object]:
        query = state.get("rewritten_query") or state["query"]
        prompt = (
            "Rewrite the user's question to be more specific and retrieval-"
            "friendly. Preserve intent and language. Return ONLY the rewritten "
            f"question, no preamble.\n\nQuestion: {query}"
        )
        request = GenerationRequest(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="",
            temperature=0.0,
        )
        result = await grade_llm.generate(request)
        return {
            "rewritten_query": result.text.strip() or query,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }

    return rewrite_query_node


def _make_retrieve_node(retrieve_uc: RetrieveUseCase) -> object:
    """Build the retrieve node bound to *retrieve_uc*."""

    @traced("agent.node.retrieve")
    async def retrieve_node(state: AgentState) -> dict[str, object]:
        query = state.get("rewritten_query") or state["query"]
        chunks = await retrieve_uc.execute(
            query=query,
            language=state.get("language"),
            session_id=state.get("session_id"),
        )
        return {"retrieved_chunks": chunks, "relevance_grades": []}

    return retrieve_node


def _make_grade_node(grade_llm: LLMPort) -> object:
    """Build the relevance-grading node bound to *grade_llm*."""

    @traced("agent.node.grade_relevance")
    async def grade_relevance_node(state: AgentState) -> dict[str, object]:
        chunks: list[RetrievedChunk] = state.get("retrieved_chunks", [])
        if not chunks:
            return {"relevance_grades": []}

        query = state.get("rewritten_query") or state["query"]

        async def _grade_one(chunk: RetrievedChunk) -> bool:
            prompt = (
                "You are a binary relevance grader. Decide whether the passage "
                "is relevant to the question. Answer with a single token: "
                "'yes' or 'no'.\n\n"
                f"Question: {query}\n\nPassage: {chunk.content}"
            )
            req = GenerationRequest(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="",
                temperature=0.0,
                max_tokens=4,
            )
            result = await grade_llm.generate(req)
            return result.text.strip().lower().startswith("y")

        grades: list[bool] = list(await asyncio.gather(*(_grade_one(c) for c in chunks)))
        return {"relevance_grades": grades}

    return grade_relevance_node


def _filter_by_grades(chunks: list[RetrievedChunk], grades: list[bool]) -> list[RetrievedChunk]:
    if grades and len(grades) == len(chunks):
        return [c for c, ok in zip(chunks, grades, strict=True) if ok]
    return chunks


def _make_generate_node(generate_uc: GenerateAnswerUseCase) -> object:
    """Build the answer-generation node bound to *generate_uc*."""

    @traced("agent.node.generate")
    async def generate_node(state: AgentState) -> dict[str, object]:
        chunks = _filter_by_grades(
            state.get("retrieved_chunks", []),
            state.get("relevance_grades", []),
        )
        answer = await generate_uc.execute(
            query=state.get("rewritten_query") or state["query"],
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
        chunks = _filter_by_grades(
            state.get("retrieved_chunks", []),
            state.get("relevance_grades", []),
        )

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


def _route_after_grade(state: AgentState) -> str:
    """Decide next node after relevance grading."""
    if any(state.get("relevance_grades", [])):
        return "generate"
    if state.get("rewrite_count", 0) == 0:
        return "rewrite_query"
    return "generate"


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

    Args:
        retrieve_uc: Injected retrieval use case.
        generate_uc: Injected answer-generation use case.
        grade_llm: LLM adapter used for rewriting, grading, and verification.
        settings: Application settings (used for loop bounds).

    Returns:
        Compiled LangGraph ready for ``ainvoke`` / ``astream``.
    """
    max_retries = settings.agent.max_regen_attempts

    sg: StateGraph = StateGraph(AgentState)
    sg.add_node("rewrite_query", _make_rewrite_node(grade_llm))
    sg.add_node("retrieve", _make_retrieve_node(retrieve_uc))
    sg.add_node("grade_relevance", _make_grade_node(grade_llm))
    sg.add_node("generate", _make_generate_node(generate_uc))
    sg.add_node("verify_grounding", _make_verify_node(grade_llm))

    sg.set_entry_point("retrieve")
    sg.add_edge("rewrite_query", "retrieve")
    sg.add_edge("retrieve", "grade_relevance")
    sg.add_conditional_edges(
        "grade_relevance",
        _route_after_grade,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    sg.add_edge("generate", "verify_grounding")
    sg.add_conditional_edges(
        "verify_grounding",
        _make_route_after_verify(max_retries),
        {"generate": "generate", END: END},
    )

    return sg.compile()
