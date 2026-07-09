"""OntoLearner AutoLearner wrapper for the ontology learning pipeline.

Integrates the five-step pipeline into OntoLearner's ``AutoLearner`` interface
so that OntoLearner's ``LearnerPipeline`` can orchestrate training, prediction,
and evaluation using our approach.

Supported task: ``"text2onto"`` (Flagship Task — end-to-end ontology learning
from raw text documents).

Input bundle format:
    fit(train_data):
        {
            "documents": [{"id": ..., "title": ..., "text": ...}, ...],
            "terms2docs": {"term": ["doc_id", ...], ...},  # gold labels
            "terms2types": {"term": ["Type", ...], ...},   # optional gold labels
        }
    predict(test_data):
        {"documents": [{"id": ..., "title": ..., "text": ...}, ...]}

Output format:
    {"terms": [{"doc_id": ..., "term": ...}, ...],
     "types": [{"doc_id": ..., "type": ...}, ...]}
"""

from __future__ import annotations

import logging
from typing import Any

from llm import LLMClient
from models import ExtractedTerm, TermTypingResult
from pipeline import Pipeline, PipelineConfig

logger = logging.getLogger(__name__)

try:
    from ontolearner.base import AutoLearner  # type: ignore[import]

    _BASE = AutoLearner
except ImportError:
    # Graceful fallback so the module can be imported without ontolearner installed
    # (useful for unit tests and standalone use).
    class _BASE:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            pass


class OntologyLearner(_BASE):
    """vLLM-backed ontology learner using the five-step pipeline.

    This learner connects to a running vLLM server and runs the full
    pipeline (S1 term extraction → S2 type extraction → S3 term typing →
    S4 taxonomy → S5 non-taxonomic RE) to produce ontology outputs from
    raw text.

    For OntoLearner integration, only ``text2onto`` is currently wired up.
    Independent task modes (``term-typing``, ``taxonomy-discovery``,
    ``non-taxonomic-re``) are available via the underlying ``Pipeline`` object.

    Args:
        base_url: vLLM server base URL (e.g. ``"http://node07:8000/v1"``).
        model: Model name as served by vLLM.
        temperature: Sampling temperature. 0.0 = deterministic.
        max_tokens: Maximum tokens per LLM call.
        k_examples: Training docs retrieved per document for RAG few-shot in S1 and S2.
        batch_size_s3: Terms per LLM call in S3.
        batch_size_s4: Candidate pairs per LLM call in S4.
        workers: Thread-pool size for parallel LLM calls.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        k_examples: int = 3,
        batch_size_s3: int = 20,
        batch_size_s4: int = 20,
        workers: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._client = LLMClient(
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        self._config = PipelineConfig(
            k_examples=k_examples,
            batch_size_s3=batch_size_s3,
            batch_size_s4=batch_size_s4,
            workers=workers,
        )
        self.pipeline = Pipeline(client=self._client, config=self._config)

        # Few-shot examples cached during fit() for injection into prompts
        self._fs_s3: list[dict[str, Any]] = []
        self._fs_s4: list[dict[str, Any]] = []
        self._fs_s5: list[dict[str, Any]] = []

    def load(self, **kwargs: Any) -> None:
        """No-op: the LLM is served externally; nothing to load locally."""

    # ------------------------------------------------------------------
    # OntoLearner interface (override fit/predict to bypass ontologizer)
    # ------------------------------------------------------------------

    def fit(
        self,
        train_data: Any,
        task: str = "text2onto",
        ontologizer: bool = False,
        **kwargs: Any,
    ) -> None:
        """Cache few-shot examples from the training split.

        For ``text2onto``: extracts (term, types) pairs from the gold
        ``terms2types`` mapping to prime S3's few-shot prompt.

        Args:
            train_data: Dict with keys ``documents``, ``terms2docs``,
                ``terms2types`` (optional).
            task: Must be ``"text2onto"`` for this learner.
            ontologizer: Ignored (set False to bypass OntoLearner's
                ``tasks_data_former`` which does not handle text2onto).
        """
        if task != "text2onto":
            raise ValueError(
                f"{self.__class__.__name__} currently supports only "
                f"task='text2onto' (got {task!r})."
            )
        terms2types: dict[str, list[str]] = train_data.get("terms2types") or {}
        self._fs_s3 = [
            {"term": term, "types": types}
            for term, types in list(terms2types.items())[:10]
            if types
        ]
        logger.info(
            "OntologyLearner.fit: cached %d few-shot examples for S3", len(self._fs_s3)
        )

    def predict(
        self,
        test_data: Any,
        task: str = "text2onto",
        ontologizer: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the pipeline on test documents and return OntoLearner predictions.

        Args:
            test_data: Dict with key ``documents``.
            task: Must be ``"text2onto"``.
            ontologizer: Ignored.

        Returns:
            ``{"terms": [{"doc_id": ..., "term": ...}, ...],
               "types": [{"doc_id": ..., "type": ...}, ...]}``
        """
        if task != "text2onto":
            raise ValueError(
                f"{self.__class__.__name__} currently supports only "
                f"task='text2onto' (got {task!r})."
            )
        documents: list[dict[str, str]] = test_data.get("documents", []) or []

        # Inject cached few-shot examples into the pipeline config
        self._config.examples_s3 = self._fs_s3

        # S1 → S2 → S3 (core of text2onto evaluation)
        terms = self.pipeline.run_s1(documents)
        types = self.pipeline.run_s2(terms)
        term_typings = self.pipeline.run_s3(terms, types)

        return _to_ontolearner_text2onto(term_typings, terms)

    def run_full_pipeline(
        self, documents: list[dict[str, str]]
    ) -> "PipelineOutput":  # noqa: F821
        """Run all five steps and return the full ``PipelineOutput``.

        Use this when you also need taxonomy and non-taxonomic relations,
        e.g. for the OntoLearner ``OntologyData`` format.

        Args:
            documents: List of ``{"id": ..., "text": ...}`` dicts.

        Returns:
            ``PipelineOutput`` convertible via ``.to_ontolearner()``.
        """
        from models import PipelineOutput  # local import to avoid cycle

        self._config.examples_s3 = self._fs_s3
        return self.pipeline.run(documents)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _to_ontolearner_text2onto(
    term_typings: list[TermTypingResult],
    terms: list[ExtractedTerm],
) -> dict[str, Any]:
    """Convert pipeline output to OntoLearner text2onto prediction format.

    Terms are linked back to their source documents using the ``source_doc_ids``
    from S1.  Types are derived from the term typings.

    Returns:
        ``{"terms": [{"doc_id": ..., "term": ...}, ...],
           "types": [{"doc_id": ..., "type": ...}, ...]}``
    """
    term_to_docs: dict[str, list[str]] = {
        t.text: t.source_doc_ids for t in terms
    }

    term_predictions: list[dict[str, str]] = []
    type_predictions: list[dict[str, str]] = []

    for tt in term_typings:
        doc_ids = term_to_docs.get(tt.term, [])
        if not doc_ids:
            doc_ids = ["unknown"]

        for doc_id in doc_ids:
            term_predictions.append({"doc_id": doc_id, "term": tt.term})
            for t in tt.types:
                type_predictions.append({"doc_id": doc_id, "type": t})

    return {"terms": term_predictions, "types": type_predictions}
