"""Step 2 — Type Extraction (per-document with RAG few-shot)."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import TYPE_CHECKING, Any

from llm import LLMClient, LLMError
from models import ExtractedType
from prompts import build_s2_per_doc_messages

if TYPE_CHECKING:
    from s0_retriever import DocRetriever

logger = logging.getLogger(__name__)
_PROMPT_PRINT_LOCK = Lock()

EXAMPLE_SELECTION_MODES = {
    "similarity",
    "random",
    "representative",
    "representative_diverse",
    "doc_then_type_similarity",
    "hybrid_similarity",
    "hybrid_score",
    "type_similarity",
}


def run(
    documents: list[dict[str, Any]],
    client: LLMClient,
    workers: int = 4,
    doc_index: DocRetriever | None = None,
    k_examples: int = 3,
    example_selection: str = "similarity",
    example_seed: int | None = None,
    cot: bool = False,
    max_consecutive_errors: int = 5,
) -> list[ExtractedType]:
    """Extract types from a corpus of documents, one document per LLM call.

    Each document is processed independently. RAG few-shot examples are
    retrieved from the training doc index (if provided).

    Args:
        documents: List of ``{"id": ..., "text": ...}`` dicts.
        client: Configured LLM client.
        workers: Maximum parallel LLM calls.
        doc_index: DocRetriever built from training data for RAG few-shot.
        k_examples: Number of similar training docs to retrieve per document.
        example_selection: Few-shot selection strategy — one of
            ``"similarity"`` (default, top-k by doc-text cosine),
            ``"random"``, ``"representative"``, ``"representative_diverse"``,
            ``"type_similarity"``, ``"doc_then_type_similarity"``,
            ``"hybrid_similarity"``, ``"hybrid_score"``.
        example_seed: Optional seed for deterministic random selection.
        max_consecutive_errors: Abort if this many consecutive docs fail.

    Returns:
        Deduplicated list of :class:`~ontology_learning.models.ExtractedType`,
        each carrying ``source_doc_ids``.
    """
    if example_selection not in EXAMPLE_SELECTION_MODES:
        raise ValueError(
            f"S2 example_selection must be one of {sorted(EXAMPLE_SELECTION_MODES)}"
            f" (got {example_selection!r})"
        )

    n_total = len(documents)
    logger.info("S2: %d documents, workers=%d, few-shot=%s",
                n_total, workers, example_selection)

    type_map: dict[str, set[str]] = {}
    errors = 0
    consecutive_errors = 0
    done = 0
    print_prompts = os.environ.get("PIPELINE_PRINT_S2_PROMPTS") == "1"

    if doc_index is not None and example_selection != "random":
        doc_index.warm()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_doc = {
            executor.submit(
                _call_doc, doc, doc_index, k_examples, client,
                print_prompts, example_selection, example_seed, cot,
            ): doc
            for doc in documents
        }
        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            done += 1
            if done % 100 == 0 or done == n_total:
                logger.info("S2 progress: %d/%d docs (%.1f%%)",
                            done, n_total, 100 * done / n_total)
            try:
                doc_types = future.result()
                consecutive_errors = 0
                for typ, doc_id in doc_types:
                    type_map.setdefault(typ, set()).add(doc_id)
            except (LLMError, Exception):
                errors += 1
                consecutive_errors += 1
                logger.error("S2: doc %s failed (errors so far: %d)",
                             doc["id"], errors, exc_info=True)
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"S2 aborted: {consecutive_errors} consecutive LLM failures"
                    )

    logger.info("S2 done: %d unique types, %d docs failed", len(type_map), errors)

    return [
        ExtractedType(text=t, source_doc_ids=sorted(type_map[t]))
        for t in sorted(type_map)
    ]


def _call_doc(
    doc: dict[str, Any],
    doc_index: DocRetriever | None,
    k_examples: int,
    client: LLMClient,
    print_prompts: bool = False,
    example_selection: str = "similarity",
    example_seed: int | None = None,
    cot: bool = False,
) -> list[tuple[str, str]]:
    doc_id = str(doc["id"])
    text = doc["text"]

    examples = _select_examples(
        doc_index=doc_index,
        doc_id=doc_id,
        text=text,
        k_examples=k_examples,
        example_selection=example_selection,
        example_seed=example_seed,
    )
    header = "Examples from similar documents:"
    messages = build_s2_per_doc_messages(text, examples=examples or None,
                                         examples_header=header, cot=cot)
    if print_prompts:
        _print_prompt(doc_id, messages)
    response = client.chat_json(messages)

    result: list[tuple[str, str]] = []
    for raw in response.get("types", []):
        typ = raw.strip().lower()
        if typ:
            result.append((typ, doc_id))
    return result


def _select_examples(
    doc_index: DocRetriever | None,
    doc_id: str,
    text: str,
    k_examples: int,
    example_selection: str,
    example_seed: int | None,
) -> list[dict]:
    if doc_index is None or k_examples <= 0:
        return []

    if example_selection == "random":
        seed = _stable_doc_seed(example_seed, doc_id) if example_seed is not None else None
        return doc_index.sample(k=k_examples, required_field="types", seed=seed)

    if example_selection == "representative":
        return doc_index.query_representative(
            text[:500], k=k_examples, pool_size=100, required_field="types")

    if example_selection == "representative_diverse":
        return doc_index.query_representative_diverse(
            text[:500], k=k_examples, pool_size=100, required_field="types")

    if example_selection == "type_similarity":
        return doc_index.query_by_type_labels(
            text[:500], k=k_examples, required_field="types")

    if example_selection == "doc_then_type_similarity":
        return doc_index.query_doc_pool_by_type_labels(
            text[:500], k=k_examples, pool_size=100, required_field="types")

    if example_selection == "hybrid_similarity":
        return _select_hybrid_similarity_examples(doc_index, text[:500], k_examples)

    if example_selection == "hybrid_score":
        return doc_index.query_hybrid_score(
            text[:500], k=k_examples, required_field="types",
            doc_weight=0.5, type_weight=0.5)

    # default: similarity — walk ranked list, skip self and empty-types docs
    examples = []
    for ex in doc_index.query(text[:500], k=len(doc_index.docs)):
        if str(ex.get("id", "")) == doc_id:
            continue
        if ex.get("types"):
            examples.append(ex)
            if len(examples) == k_examples:
                break
    return examples


def _select_hybrid_similarity_examples(
    doc_index: DocRetriever,
    text: str,
    k_examples: int,
) -> list[dict]:
    selected: list[dict] = []
    selected_keys: set[str] = set()

    for ex in doc_index.query(text, k=k_examples, required_field="types"):
        _append_unique(selected, selected_keys, ex)

    label_candidates = doc_index.query_by_type_labels(
        text, k=k_examples + len(selected), required_field="types")

    added = 0
    for ex in label_candidates:
        if _append_unique(selected, selected_keys, ex):
            added += 1
        if added == k_examples:
            break

    return selected


def _append_unique(selected: list[dict], keys: set[str], ex: dict) -> bool:
    key = str(ex.get("id", ex.get("text", id(ex))))
    if key in keys:
        return False
    selected.append(ex)
    keys.add(key)
    return True


def _stable_doc_seed(base_seed: int, doc_id: str) -> int:
    digest = hashlib.blake2b(
        f"{base_seed}:{doc_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _print_prompt(doc_id: str, messages: list[dict[str, str]]) -> None:
    with _PROMPT_PRINT_LOCK:
        print(f"\n===== S2 PROMPT START doc_id={doc_id} =====", file=sys.stderr)
        for i, msg in enumerate(messages):
            print(f"\n--- message[{i}] role={msg.get('role','')} ---", file=sys.stderr)
            print(msg.get("content", ""), file=sys.stderr)
        print(f"===== S2 PROMPT END doc_id={doc_id} =====\n", file=sys.stderr, flush=True)
