"""Step 1 — Term Extraction (per-document with RAG few-shot)."""

from __future__ import annotations

import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from llm import LLMClient, LLMError
from models import ExtractedTerm
from prompts import build_s1_per_doc_messages

if TYPE_CHECKING:
    from s0_retriever import DocRetriever

logger = logging.getLogger(__name__)

EXAMPLE_SELECTION_MODES = {
    "similarity_all_docs",   # top-k from all docs regardless of annotation (Round 1)
    "similarity_two_pools",  # k from term-bearing + k_empty from type-only (Round 2)
    "random",                # random from term-bearing docs (baseline)
}


def load_classifier(classifier_path: str):
    """Load a pre-trained term classifier from disk (optional pre-filter)."""
    import pickle
    with open(classifier_path, "rb") as f:
        obj = pickle.load(f)
    return obj["pipeline"], obj["threshold"]


def run(
    documents: list[dict[str, Any]],
    client: LLMClient,
    workers: int = 4,
    doc_index: DocRetriever | None = None,
    k_examples: int = 3,
    example_selection: str = "similarity_all_docs",
    example_seed: int | None = None,
    k_empty: int = 0,
    classifier_path: str | None = None,
    cot: bool = False,
    max_consecutive_errors: int = 5,
) -> list[ExtractedTerm]:
    """Extract terms from a corpus of documents, one document per LLM call.

    Args:
        documents: List of ``{"id": ..., "text": ...}`` dicts.
        client: Configured LLM client.
        workers: Maximum parallel LLM calls.
        doc_index: DocRetriever built from training data for RAG few-shot.
        k_examples: Number of examples from the primary pool.
        example_selection: How to select ``k_examples``:
            - ``"similarity_all_docs"`` (default) — top-k by similarity from all
              training docs regardless of annotation; auto-split for mixed prompt.
            - ``"similarity_two_pools"`` — k from term-bearing docs + k_empty from
              type-only docs (two explicit pools; Round 2+).
            - ``"random"`` — random sample from term-bearing docs.
        example_seed: Seed for deterministic random selection.
        k_empty: Number of additional type-only examples from the secondary pool.
            Only used when ``example_selection="similarity_two_pools"``.
        classifier_path: Path to term classifier pkl — docs predicted as type-only
            are skipped without an LLM call.
        max_consecutive_errors: Abort if this many consecutive docs fail.
    """
    if example_selection not in EXAMPLE_SELECTION_MODES:
        raise ValueError(
            f"S1 example_selection must be one of {sorted(EXAMPLE_SELECTION_MODES)}"
            f" (got {example_selection!r})"
        )

    # ── Optional classifier pre-filter ───────────────────────────────────────
    if classifier_path:
        clf_pipeline, clf_threshold = load_classifier(classifier_path)
        proba = clf_pipeline.predict_proba([doc["text"] for doc in documents])[:, 1]
        filtered = [doc for doc, p in zip(documents, proba) if p >= clf_threshold]
        skipped = len(documents) - len(filtered)
        logger.info("S1 classifier: %d/%d docs pass filter (%.1f%% skipped)",
                    len(filtered), len(documents), 100 * skipped / len(documents))
        documents = filtered

    n_total = len(documents)
    logger.info("S1: %d documents to process, workers=%d, selection=%s",
                n_total, workers, example_selection)

    if doc_index is not None and example_selection != "random":
        doc_index.warm()

    term_map: dict[str, dict] = {}
    errors = 0
    consecutive_errors = 0
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_doc = {
            executor.submit(
                _call_doc, doc, doc_index, k_examples, client,
                example_selection, example_seed, k_empty, cot,
            ): doc
            for doc in documents
        }
        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            done += 1
            if done % 100 == 0 or done == n_total:
                logger.info("S1 progress: %d/%d docs (%.1f%%)",
                            done, n_total, 100 * done / n_total)
            try:
                doc_terms = future.result()
                consecutive_errors = 0
                for term_text, data in doc_terms.items():
                    if term_text not in term_map:
                        term_map[term_text] = {
                            "doc_ids": set(data["doc_ids"]),
                            "context": data["context"],
                        }
                    else:
                        term_map[term_text]["doc_ids"].update(data["doc_ids"])
            except (LLMError, Exception):
                errors += 1
                consecutive_errors += 1
                logger.error("S1: doc %s failed (errors so far: %d)",
                             doc["id"], errors, exc_info=True)
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"S1 aborted: {consecutive_errors} consecutive LLM failures"
                    )

    logger.info("S1 done: %d unique terms, %d docs failed", len(term_map), errors)
    return [
        ExtractedTerm(
            text=t,
            source_doc_ids=sorted(term_map[t]["doc_ids"]),
            context_sentence=term_map[t]["context"],
        )
        for t in sorted(term_map)
    ]


def _call_doc(
    doc: dict[str, Any],
    doc_index: DocRetriever | None,
    k_examples: int,
    client: LLMClient,
    example_selection: str = "similarity_all_docs",
    example_seed: int | None = None,
    k_empty: int = 0,
    cot: bool = False,
) -> dict[str, dict]:
    doc_id = str(doc["id"])
    text = doc["text"]

    examples, examples_empty = _select_examples(
        doc_index, doc_id, text, k_examples, example_selection, example_seed, k_empty)

    messages = build_s1_per_doc_messages(
        text,
        examples=examples or None,
        examples_header="Examples from similar documents:",
        cot=cot,
        examples_empty=examples_empty if examples_empty else None,
    )
    response = client.chat_json(messages)

    result: dict[str, dict] = {}
    for raw in response.get("terms", []):
        term = raw.strip().lower()
        if not term:
            continue
        if term not in result:
            result[term] = {
                "doc_ids": [doc_id],
                "context": _find_context_sentence(term, text),
            }
    return result


def _select_examples(
    doc_index: DocRetriever | None,
    doc_id: str,
    text: str,
    k_examples: int,
    example_selection: str,
    example_seed: int | None,
    k_empty: int,
) -> tuple[list[dict], list[dict]]:
    """Return (examples_with_terms, examples_without_terms)."""
    if doc_index is None or k_examples <= 0:
        return [], []

    candidates = [
        ex for ex in doc_index.query(text[:500], k=len(doc_index.docs))
        if str(ex.get("id", "")) != doc_id
    ]

    if example_selection == "similarity_all_docs":
        # Top-k from all docs — auto-split for mixed prompt; k_empty ignored
        selected = candidates[:k_examples]
        return ([ex for ex in selected if ex.get("terms")],
                [ex for ex in selected if not ex.get("terms")])

    if example_selection == "random":
        import random
        seed = _stable_seed(example_seed, doc_id) if example_seed is not None else None
        term_candidates = [ex for ex in candidates if ex.get("terms")]
        rng = random.Random(seed)
        examples = rng.sample(term_candidates, min(k_examples, len(term_candidates)))
    else:
        # "similarity_two_pools" — top-k from term-bearing docs
        examples = []
        for ex in candidates:
            if ex.get("terms"):
                examples.append(ex)
                if len(examples) == k_examples:
                    break

    # Secondary type-only pool (similarity_two_pools and random)
    examples_empty: list[dict] = []
    if k_empty > 0:
        seen = {id(ex) for ex in examples}
        for ex in candidates:
            if not ex.get("terms") and id(ex) not in seen:
                examples_empty.append(ex)
                if len(examples_empty) == k_empty:
                    break

    return examples, examples_empty


def _stable_seed(base_seed: int, doc_id: str) -> int:
    digest = hashlib.blake2b(
        f"{base_seed}:{doc_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _find_context_sentence(term: str, text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    term_lower = term.lower()
    for sent in sentences:
        if term_lower in sent.lower():
            return sent.strip()
    return text[:200].strip()
