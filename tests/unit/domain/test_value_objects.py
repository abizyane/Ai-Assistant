"""Tests for domain value objects."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.domain.value_objects.embedding import Embedding
from src.domain.value_objects.language import Language
from src.domain.value_objects.score import Score, ScoreKind


class TestEmbedding:
    def test_embedding_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValidationError, match="vector length"):
            Embedding(vector=(0.1, 0.2, 0.3), dimension=4, model="bge-m3")

    def test_embedding_constructs_when_dimension_matches(self) -> None:
        emb = Embedding(vector=(0.1, 0.2, 0.3), dimension=3, model="bge-m3")
        assert len(emb.vector) == 3
        assert emb.dimension == 3

    def test_embedding_frozen_mutation_raises(self) -> None:
        emb = Embedding(vector=(0.1,), dimension=1, model="bge-m3")
        with pytest.raises(ValidationError):
            emb.dimension = 2  # type: ignore[misc]

    def test_embedding_empty_vector_raises(self) -> None:
        with pytest.raises(ValidationError):
            Embedding(vector=(), dimension=0, model="bge-m3")

    def test_embedding_large_vector(self) -> None:
        vec = tuple(float(i) / 1024 for i in range(1024))
        emb = Embedding(vector=vec, dimension=1024, model="bge-m3")
        assert emb.dimension == 1024


class TestScore:
    def test_score_out_of_range_above_raises(self) -> None:
        with pytest.raises(ValidationError):
            Score(value=1.1, kind=ScoreKind.DENSE)

    def test_score_out_of_range_below_raises(self) -> None:
        with pytest.raises(ValidationError):
            Score(value=-0.01, kind=ScoreKind.RERANK)

    def test_score_constructs_at_boundaries(self) -> None:
        s_min = Score(value=0.0, kind=ScoreKind.FAITHFULNESS)
        s_max = Score(value=1.0, kind=ScoreKind.RELEVANCE)
        assert s_min.value == pytest.approx(0.0)
        assert s_max.value == pytest.approx(1.0)

    def test_score_frozen_mutation_raises(self) -> None:
        s = Score(value=0.5, kind=ScoreKind.SPARSE)
        with pytest.raises(ValidationError):
            s.value = 0.9  # type: ignore[misc]

    def test_score_kind_values(self) -> None:
        assert ScoreKind.DENSE == "dense"
        assert ScoreKind.FAITHFULNESS == "faithfulness"


class TestLanguage:
    def test_language_valid_code_normalizes_to_lowercase(self) -> None:
        lang = Language(code="EN")
        assert lang.code == "en"

    def test_language_valid_fr(self) -> None:
        lang = Language(code="fr")
        assert lang.code == "fr"

    def test_language_valid_ar(self) -> None:
        lang = Language(code="ar")
        assert lang.code == "ar"

    def test_language_invalid_code_raises(self) -> None:
        with pytest.raises(ValidationError, match="Unknown ISO-639-1"):
            Language(code="xx")

    def test_language_frozen_mutation_raises(self) -> None:
        lang = Language(code="en")
        with pytest.raises(ValidationError):
            lang.code = "fr"  # type: ignore[misc]

    def test_language_wrong_length_raises(self) -> None:
        with pytest.raises(ValidationError):
            Language(code="eng")

    def test_language_empty_raises(self) -> None:
        with pytest.raises(ValidationError):
            Language(code="")
