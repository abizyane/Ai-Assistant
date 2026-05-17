"""Generate-answer use case — turns retrieved chunks into a grounded answer."""
from __future__ import annotations

import logging
import re
from typing import Final

from src.application.prompts import PromptTemplateLoader
from src.domain.entities.answer import AnswerCitation, AnswerWithCitations
from src.domain.entities.session import Message
from src.domain.ports.dto import GenerationRequest, RetrievedChunk
from src.domain.ports.llm import LLMPort
from src.domain.ports.tracer import TracerPort
from src.shared.metrics import inc_counter, observe_histogram
from src.shared.tracing import traced

__all__ = ["GenerateAnswerUseCase"]

_TEMPLATE_NAME: Final = "answer_with_citations.j2"

_CITATION_MARKER_RE: Final = re.compile(r"\[([a-zA-Z0-9_\-:]+)\]")

_REFUSAL_PHRASES: Final[dict[str, str]] = {
    "en": (
        "I don't have enough information to answer that based on the 1337 "
        "Coding School knowledge base."
    ),
    "fr": (
        "Je ne dispose pas d'informations suffisantes pour répondre à cette "
        "question à partir de la base de connaissances de 1337."
    ),
    "ar": "لا تتوفر لدي معلومات كافية للإجابة على هذا السؤال من قاعدة معارف مدرسة 1337.",
}


def _refusal_for(language: str) -> str:
    return _REFUSAL_PHRASES.get(language.lower(), _REFUSAL_PHRASES["en"])


class GenerateAnswerUseCase:
    """Generate a grounded, multilingual answer from retrieved chunks.

    Behaviour:
        * Empty ``retrieved_chunks`` short-circuits to a localized refusal
          without invoking the LLM (deterministic, anti-hallucination).
        * The LLM is instructed to cite chunks using ``[<chunk_id>]`` markers.
        * After generation, markers are parsed and any that do NOT correspond
          to a retrieved chunk are dropped (no fabricated sources).
    """

    def __init__(
        self,
        llm: LLMPort,
        prompt_template_loader: PromptTemplateLoader,
        tracer: TracerPort | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Wire collaborators via constructor injection.

        Args:
            llm: Adapter implementing ``LLMPort`` for text generation.
            prompt_template_loader: Loader for Jinja2 prompt templates.
            tracer: Optional tracing adapter (currently unused at this level
                — the ``@traced`` decorator owns OTel spans).
            logger: Optional logger for error/warn diagnostics.
        """
        self._llm = llm
        self._prompts = prompt_template_loader
        self._tracer = tracer
        self._log = logger or logging.getLogger(__name__)

    @traced("generate_answer.execute")
    async def execute(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        history: list[Message],
        language: str,
    ) -> AnswerWithCitations:
        """Generate an answer for ``query`` grounded in ``retrieved_chunks``.

        Args:
            query: The user question.
            retrieved_chunks: Top-K chunks selected by retrieval/reranking.
            history: Prior conversation turns (oldest first).
            language: BCP-47 language code (e.g. ``en``, ``fr``, ``ar``).

        Returns:
            ``AnswerWithCitations`` carrying answer text, grounded citations,
            and token usage.
        """
        if not retrieved_chunks:
            return AnswerWithCitations(
                text=_refusal_for(language),
                citations=[],
                language=language,
                tokens_in=0,
                tokens_out=0,
            )

        prompt = self._prompts.render(
            _TEMPLATE_NAME,
            query=query,
            chunks=[self._chunk_view(c) for c in retrieved_chunks],
            history=[{"role": m.role.value, "content": m.content} for m in history],
            language=language,
            refusal_phrase=_refusal_for(language),
        )

        request = GenerationRequest(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="",
            temperature=0.2,
        )

        try:
            result = await self._llm.generate(request)
        except Exception:
            self._log.exception("generate_answer.llm_failed", extra={"language": language})
            raise

        citations = self._extract_grounded_citations(result.text, retrieved_chunks)

        inc_counter("generation_tokens_total", {"kind": "input"})
        inc_counter("generation_tokens_total", {"kind": "output"})
        observe_histogram("generation_citations_count", float(len(citations)))

        return AnswerWithCitations(
            text=result.text,
            citations=citations,
            language=language,
            tokens_in=result.input_tokens,
            tokens_out=result.output_tokens,
        )

    @staticmethod
    def _chunk_view(chunk: RetrievedChunk) -> dict[str, object]:
        return {
            "chunk_id": str(chunk.chunk_id),
            "source_path": chunk.source_path,
            "page": chunk.metadata.get("page"),
            "content": chunk.content,
        }

    @staticmethod
    def _extract_grounded_citations(
        text: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[AnswerCitation]:
        by_id = {str(c.chunk_id): c for c in retrieved_chunks}
        seen: set[str] = set()
        citations: list[AnswerCitation] = []
        for match in _CITATION_MARKER_RE.finditer(text):
            chunk_id = match.group(1)
            if chunk_id in seen or chunk_id not in by_id:
                continue
            seen.add(chunk_id)
            chunk = by_id[chunk_id]
            page = chunk.metadata.get("page")
            citations.append(
                AnswerCitation(
                    chunk_id=chunk_id,
                    source=chunk.source_path,
                    page=int(page) if isinstance(page, int) else None,
                    marker=match.group(0),
                )
            )
        return citations
