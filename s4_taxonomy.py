"""Step 4 — Taxonomy Discovery (LLM verification or embedding similarity).

Processes per-document: for each document, discovers is-a relationships among
the types assigned to its terms in S3. This matches the challenge format where
each document's primitive-ontology-triples are evaluated independently.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import sys
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from s0_retriever import DocRetriever, TaxonomyParentIndex

from llm import LLMClient, LLMError, _extract_json
from models import ExtractedType, TaxonomyResult, TermTypingResult
from progress import progress_bar
from prompts import build_s4_cluster_messages, build_s4_depth_messages, build_s4_messages

logger = logging.getLogger(__name__)
S4_MODES = {
    "labels",
    "boolean",
    "fewshot_pos",
    "fewshot_pos_neg",
    "fewshot_grouped_pos_neg",
    "fewshot_grouped_pos_neg_child_batches",
    "fewshot_grouped_pos_neg_lexical",
    "fewshot_grouped_pos_neg_lexical_inverse_pruned",
    "fewshot_grouped_pos_neg_lexical_child_batches",
    "fewshot_grouped_pos_neg_lexical_child_batches_single_parent",
}
_GROUPED_FEWSHOT_MODES = {
    "fewshot_grouped_pos_neg",
    "fewshot_grouped_pos_neg_child_batches",
    "fewshot_grouped_pos_neg_lexical",
    "fewshot_grouped_pos_neg_lexical_inverse_pruned",
    "fewshot_grouped_pos_neg_lexical_child_batches",
    "fewshot_grouped_pos_neg_lexical_child_batches_single_parent",
}
_FEWSHOT_MODES = {
    "fewshot_pos",
    "fewshot_pos_neg",
    *_GROUPED_FEWSHOT_MODES,
}
_LEXICAL_SHORTCUT_MODES = {
    "fewshot_grouped_pos_neg_lexical",
    "fewshot_grouped_pos_neg_lexical_inverse_pruned",
    "fewshot_grouped_pos_neg_lexical_child_batches",
    "fewshot_grouped_pos_neg_lexical_child_batches_single_parent",
}
_LEXICAL_INVERSE_PRUNING_MODES = {
    "fewshot_grouped_pos_neg_lexical_inverse_pruned",
}
_CHILD_GROUPED_BATCH_MODES = {
    "fewshot_grouped_pos_neg_child_batches",
    "fewshot_grouped_pos_neg_lexical_child_batches",
    "fewshot_grouped_pos_neg_lexical_child_batches_single_parent",
}
_SINGLE_PARENT_PROMPT_MODES = {
    "fewshot_grouped_pos_neg_lexical_child_batches_single_parent",
}
_DEBUG_PRINT_LOCK = Lock()


@dataclass(frozen=True)
class _DocCandidates:
    doc_id: str
    type_vocab: list[str]
    candidates: list[tuple[str, str]]


@dataclass(frozen=True)
class _BatchTask:
    doc_id: str
    batch_index: int
    pairs: list[tuple[str, str]]
    examples: list[dict[str, object]]


def run(
    term_typings: list[TermTypingResult],
    client: LLMClient,
    batch_size: int = 20,
    workers: int = 4,
    max_consecutive_errors: int = 5,
    depth_pass: bool = False,
    cluster_types: bool = False,
    n_clusters: int = 10,
    mode: str = "labels",
    doc_index: DocRetriever | None = None,
    documents: list[dict[str, str]] | None = None,
    k_examples: int = 3,
    max_positive_examples: int = 20,
    max_negative_examples: int = 20,
    example_seed: int | None = None,
) -> list[TaxonomyResult]:
    """Discover is-a relationships among types, per document.

    For each document, collects the types assigned to its terms (from S3) and
    runs is-a discovery only among those types. Results carry the document's ID
    directly — no reverse mapping needed at assembly time.

    Args:
        term_typings: Output of S3 (each entry has source_doc_ids).
        client: Configured LLM client.
        batch_size: Number of candidate pairs per LLM call.
        workers: Number of concurrent S4 LLM batch calls.
        max_consecutive_errors: Abort if this many S4 LLM batches fail in a row.
        depth_pass: Whether to run the depth re-prompt second pass.
        cluster_types: Enable semantic clustering before candidate generation.
        n_clusters: Target cluster count (hint to LLM, not enforced).
        mode: S4 prompting mode: ``labels`` (current), ``boolean``,
            ``fewshot_pos``, ``fewshot_pos_neg``, or
            ``fewshot_grouped_pos_neg``. The
            ``fewshot_grouped_pos_neg_lexical`` variant first auto-confirms
            candidates whose parent label is lexically included in the child.
            The ``fewshot_grouped_pos_neg_lexical_inverse_pruned`` variant also
            rejects the inverse candidate without sending it to the LLM.
            The ``fewshot_grouped_pos_neg_child_batches`` variant groups all
            candidates by child before batching, without lexical auto-confirm.
            The ``fewshot_grouped_pos_neg_lexical_child_batches`` variant
            groups remaining candidates by child after lexical auto-confirm.
            The ``fewshot_grouped_pos_neg_lexical_child_batches_single_parent``
            variant additionally asks the LLM to select at most one parent per
            child.
        doc_index: Training document retriever used by few-shot modes.
        documents: Target corpus documents, needed by few-shot modes for
            doc-to-doc similarity.
        k_examples: Number of similar training documents for S4 few-shot modes.
        max_positive_examples: Max positive is-a examples injected per target doc.
        max_negative_examples: Max generated negative examples injected per target doc.
        example_seed: Optional seed for deterministic negative example sampling.

    Returns:
        List of ``TaxonomyResult`` is-a pairs, each tagged with its source doc.
    """
    doc_to_types = _doc_to_types_from_term_typings(term_typings)
    return _run_doc_type_map(
        doc_to_types=doc_to_types,
        client=client,
        batch_size=batch_size,
        workers=workers,
        max_consecutive_errors=max_consecutive_errors,
        depth_pass=depth_pass,
        cluster_types=cluster_types,
        n_clusters=n_clusters,
        mode=mode,
        doc_index=doc_index,
        documents=documents,
        k_examples=k_examples,
        max_positive_examples=max_positive_examples,
        max_negative_examples=max_negative_examples,
        example_seed=example_seed,
    )


def run_from_types(
    types: list[ExtractedType],
    client: LLMClient,
    batch_size: int = 20,
    workers: int = 4,
    max_consecutive_errors: int = 5,
    depth_pass: bool = False,
    cluster_types: bool = False,
    n_clusters: int = 10,
    mode: str = "labels",
    doc_index: DocRetriever | None = None,
    documents: list[dict[str, str]] | None = None,
    k_examples: int = 3,
    max_positive_examples: int = 20,
    max_negative_examples: int = 20,
    example_seed: int | None = None,
) -> list[TaxonomyResult]:
    """Discover is-a relationships among S2 extracted types, per document."""
    doc_to_types = _doc_to_types_from_extracted_types(types)
    return _run_doc_type_map(
        doc_to_types=doc_to_types,
        client=client,
        batch_size=batch_size,
        workers=workers,
        max_consecutive_errors=max_consecutive_errors,
        depth_pass=depth_pass,
        cluster_types=cluster_types,
        n_clusters=n_clusters,
        mode=mode,
        doc_index=doc_index,
        documents=documents,
        k_examples=k_examples,
        max_positive_examples=max_positive_examples,
        max_negative_examples=max_negative_examples,
        example_seed=example_seed,
    )


def _doc_to_types_from_term_typings(
    term_typings: list[TermTypingResult],
) -> dict[str, set[str]]:
    doc_to_types: dict[str, set[str]] = defaultdict(set)
    for tt in term_typings:
        for doc_id in tt.source_doc_ids:
            doc_to_types[doc_id].update(tt.types)
    return doc_to_types


def _doc_to_types_from_extracted_types(
    types: list[ExtractedType],
) -> dict[str, set[str]]:
    doc_to_types: dict[str, set[str]] = defaultdict(set)
    for typ in types:
        text = typ.text.strip()
        if not text:
            continue
        for doc_id in typ.source_doc_ids:
            doc_to_types[doc_id].add(text)
    return doc_to_types


def _normalize_mode(mode: str) -> str:
    aliases = {
        "current": "labels",
        "label": "labels",
        "relation_labels": "labels",
        "simple": "boolean",
        "true_false": "boolean",
        "fewshot": "fewshot_pos",
        "fewshot_positive": "fewshot_pos",
        "fewshot_positive_negative": "fewshot_pos_neg",
        "fewshot_grouped": "fewshot_grouped_pos_neg",
        "fewshot_contrastive": "fewshot_grouped_pos_neg",
        "fewshot_child_batches": "fewshot_grouped_pos_neg_child_batches",
        "fewshot_grouped_child_batches": "fewshot_grouped_pos_neg_child_batches",
        "fewshot_grouped_by_child_no_lexical": "fewshot_grouped_pos_neg_child_batches",
        "fewshot_lexical": "fewshot_grouped_pos_neg_lexical",
        "fewshot_grouped_lexical": "fewshot_grouped_pos_neg_lexical",
        "fewshot_lexical_inverse": "fewshot_grouped_pos_neg_lexical_inverse_pruned",
        "fewshot_lexical_pruned": "fewshot_grouped_pos_neg_lexical_inverse_pruned",
        "fewshot_lexical_child_batches": "fewshot_grouped_pos_neg_lexical_child_batches",
        "fewshot_grouped_by_child": "fewshot_grouped_pos_neg_lexical_child_batches",
        "fewshot_single_parent": (
            "fewshot_grouped_pos_neg_lexical_child_batches_single_parent"
        ),
    }
    normalized = aliases.get(mode.strip().lower(), mode.strip().lower())
    if normalized not in S4_MODES:
        raise ValueError(f"S4 mode must be one of {sorted(S4_MODES)} (got {mode!r})")
    return normalized


def _doc_texts(documents: list[dict[str, str]] | None) -> dict[str, str]:
    if not documents:
        return {}
    return {
        str(doc["id"]): str(doc.get("text", ""))
        for doc in documents
        if doc.get("id") is not None
    }


def _run_doc_type_map(
    doc_to_types: dict[str, set[str]],
    client: LLMClient,
    batch_size: int,
    workers: int,
    max_consecutive_errors: int,
    depth_pass: bool,
    cluster_types: bool,
    n_clusters: int,
    mode: str,
    doc_index: DocRetriever | None,
    documents: list[dict[str, str]] | None,
    k_examples: int,
    max_positive_examples: int,
    max_negative_examples: int,
    example_seed: int | None,
) -> list[TaxonomyResult]:
    if batch_size <= 0:
        raise ValueError(f"S4 batch_size must be positive, got {batch_size}")

    mode = _normalize_mode(mode)
    worker_count = max(1, workers)
    docs_with_pairs = {doc_id: sorted(types) for doc_id, types in doc_to_types.items()
                       if len(types) >= 2}
    n_candidate_pairs = sum(len(types) * (len(types) - 1) for types in docs_with_pairs.values())
    n_batches = (n_candidate_pairs + batch_size - 1) // batch_size if batch_size > 0 else 0
    max_types = max((len(types) for types in docs_with_pairs.values()), default=0)
    avg_types = (
        sum(len(types) for types in docs_with_pairs.values()) / len(docs_with_pairs)
        if docs_with_pairs else 0.0
    )
    logger.info(
        "S4: %d documents, %d have ≥2 types for taxonomy discovery",
        len(doc_to_types), len(docs_with_pairs),
    )
    logger.info(
        "S4: %.1f avg types/doc, %d max types/doc, up to %d candidate pairs "
        "(~%d LLM batches before clustering, batch_size=%d)",
        avg_types, max_types, n_candidate_pairs, n_batches, batch_size,
    )

    doc_candidates = _prepare_doc_candidates(
        docs_with_pairs=docs_with_pairs,
        client=client,
        cluster_types=cluster_types,
        n_clusters=n_clusters,
        workers=worker_count,
        max_consecutive_errors=max_consecutive_errors,
    )
    auto_confirmed_by_doc, llm_doc_candidates = _apply_lexical_shortcut(
        doc_candidates,
        mode,
    )
    examples_by_doc = _select_examples_by_doc(
        doc_candidates=llm_doc_candidates,
        documents=documents,
        doc_index=doc_index,
        mode=mode,
        k_examples=k_examples,
        max_positive_examples=max_positive_examples,
        max_negative_examples=max_negative_examples,
        example_seed=example_seed,
    )
    batch_tasks = _make_batch_tasks(
        llm_doc_candidates,
        batch_size,
        examples_by_doc,
        mode=mode,
    )
    if mode in _CHILD_GROUPED_BATCH_MODES:
        if mode in _SINGLE_PARENT_PROMPT_MODES:
            logger.info(
                "S4 batching: candidates grouped by child; oversized child "
                "groups stay intact so the LLM can select one parent "
                "(configured batch_size=%d)",
                batch_size,
            )
        else:
            logger.info(
                "S4 batching: candidates grouped by child; groups larger than "
                "batch_size=%d are split",
                batch_size,
            )
    logger.info(
        "S4: processing %d LLM batches with %d worker(s)",
        len(batch_tasks), worker_count,
    )

    confirmed_by_doc: dict[str, list[TaxonomyResult]] = defaultdict(list)
    for doc_id, relations in auto_confirmed_by_doc.items():
        confirmed_by_doc[doc_id].extend(relations)
    batch_errors = 0
    consecutive_errors = 0
    if batch_tasks:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_task = {
                executor.submit(
                    _call_batch,
                    task.pairs,
                    client,
                    mode=mode,
                    examples=task.examples,
                    call_label=(
                        f"classification doc={task.doc_id} "
                        f"batch={task.batch_index} mode={mode}"
                    ),
                ): task
                for task in batch_tasks
            }
            for future in progress_bar(
                as_completed(future_to_task),
                total=len(future_to_task),
                desc="S4 batches",
                unit="batch",
            ):
                task = future_to_task[future]
                try:
                    confirmed_by_doc[task.doc_id].extend(future.result())
                    consecutive_errors = 0
                except (LLMError, Exception):
                    batch_errors += 1
                    consecutive_errors += 1
                    logger.warning(
                        "S4 doc %s: batch %d failed (batch errors so far: %d)",
                        task.doc_id, task.batch_index, batch_errors, exc_info=True,
                    )
                    if consecutive_errors >= max_consecutive_errors:
                        raise RuntimeError(
                            f"S4 aborted: {consecutive_errors} consecutive batch failures"
                        )

    results_by_doc: dict[str, list[TaxonomyResult]] = {}
    for doc in doc_candidates:
        results_by_doc[doc.doc_id] = _dedupe_doc_results(
            doc.doc_id, confirmed_by_doc.get(doc.doc_id, []),
        )

    if depth_pass:
        _run_depth_passes(doc_candidates, results_by_doc, client, worker_count)

    results: list[TaxonomyResult] = []
    for doc in doc_candidates:
        results.extend(results_by_doc.get(doc.doc_id, []))

    logger.info(
        "S4 done: %d is-a relations total, %d batches failed",
        len(results), batch_errors,
    )
    return results


def _lexical_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


def _is_parent_lexically_included(parent: str, child: str) -> bool:
    parent_tokens = _lexical_tokens(parent)
    child_tokens = _lexical_tokens(child)
    if not parent_tokens or len(child_tokens) <= len(parent_tokens):
        return False

    width = len(parent_tokens)
    return any(
        child_tokens[start:start + width] == parent_tokens
        for start in range(len(child_tokens) - width + 1)
    )


def _apply_lexical_shortcut(
    doc_candidates: list[_DocCandidates],
    mode: str,
) -> tuple[dict[str, list[TaxonomyResult]], list[_DocCandidates]]:
    if mode not in _LEXICAL_SHORTCUT_MODES:
        return {}, doc_candidates

    confirmed_by_doc: dict[str, list[TaxonomyResult]] = defaultdict(list)
    llm_doc_candidates: list[_DocCandidates] = []
    total_candidates = 0
    auto_confirmed = 0
    auto_rejected_inverse = 0

    for doc in doc_candidates:
        lexical_pairs = {
            (parent, child)
            for parent, child in doc.candidates
            if _is_parent_lexically_included(parent, child)
        }
        inverse_pairs = (
            {(child, parent) for parent, child in lexical_pairs}
            if mode in _LEXICAL_INVERSE_PRUNING_MODES
            else set()
        )
        remaining: list[tuple[str, str]] = []
        total_candidates += len(doc.candidates)
        for parent, child in doc.candidates:
            pair = (parent, child)
            if pair in lexical_pairs:
                confirmed_by_doc[doc.doc_id].append(
                    TaxonomyResult(parent=parent, child=child)
                )
                auto_confirmed += 1
            elif pair in inverse_pairs:
                auto_rejected_inverse += 1
            else:
                remaining.append(pair)

        if remaining:
            llm_doc_candidates.append(_DocCandidates(
                doc_id=doc.doc_id,
                type_vocab=doc.type_vocab,
                candidates=remaining,
            ))

    if mode in _LEXICAL_INVERSE_PRUNING_MODES:
        logger.info(
            "S4 lexical shortcut: %d/%d candidate pairs auto-confirmed, "
            "%d inverse pairs auto-rejected; %d pairs remain for the LLM",
            auto_confirmed,
            total_candidates,
            auto_rejected_inverse,
            total_candidates - auto_confirmed - auto_rejected_inverse,
        )
    else:
        logger.info(
            "S4 lexical shortcut: %d/%d candidate pairs auto-confirmed; "
            "%d pairs remain for the LLM",
            auto_confirmed,
            total_candidates,
            total_candidates - auto_confirmed,
        )
    return dict(confirmed_by_doc), llm_doc_candidates


def _select_examples_by_doc(
    doc_candidates: list[_DocCandidates],
    documents: list[dict[str, str]] | None,
    doc_index: DocRetriever | None,
    mode: str,
    k_examples: int,
    max_positive_examples: int,
    max_negative_examples: int,
    example_seed: int | None,
) -> dict[str, list[dict[str, object]]]:
    if mode not in _FEWSHOT_MODES:
        return {}
    if doc_index is None:
        raise ValueError(f"S4 mode {mode!r} requires a training document index")

    texts = _doc_texts(documents)
    if not texts:
        raise ValueError(f"S4 mode {mode!r} requires target corpus documents")

    doc_index.warm()
    if not any(doc.get("taxonomic_pairs") for doc in doc_index.docs):
        logger.warning(
            "S4 few-shot mode requested, but the loaded S0 doc index has no "
            "taxonomic_pairs. Rebuild S0 to enable S4 examples."
        )

    gold_pairs = _gold_pair_set(doc_index.docs)
    examples_by_doc: dict[str, list[dict[str, object]]] = {}
    for doc in progress_bar(
        doc_candidates,
        total=len(doc_candidates),
        desc="S4 examples",
        unit="doc",
    ):
        text = texts.get(doc.doc_id, "")
        if not text:
            logger.warning("S4 doc %s: no corpus text; no few-shot examples", doc.doc_id)
            examples_by_doc[doc.doc_id] = []
            continue

        similar_docs = doc_index.query(
            text[:500],
            k=k_examples,
            required_field="taxonomic_pairs",
        )
        all_positive = _positive_examples_from_docs(similar_docs, max_examples=None)
        positive = all_positive[:max(0, max_positive_examples)]
        examples = list(positive)
        if mode in {"fewshot_pos_neg", *_GROUPED_FEWSHOT_MODES}:
            negative_source = (
                positive
                if mode in _GROUPED_FEWSHOT_MODES
                else all_positive
            )
            examples.extend(
                _negative_examples_from_positive_pairs(
                    negative_source,
                    gold_pairs=gold_pairs,
                    max_examples=max_negative_examples,
                    seed=_stable_seed(example_seed, doc.doc_id),
                )
            )
        examples_by_doc[doc.doc_id] = examples

    n_examples = sum(len(v) for v in examples_by_doc.values())
    logger.info("S4: selected %d few-shot pair examples across %d docs", n_examples,
                len(examples_by_doc))
    return examples_by_doc


def _gold_pair_set(docs: list[dict]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for doc in docs:
        for pair in doc.get("taxonomic_pairs", []) or []:
            parent = str(pair.get("parent", "")).strip()
            child = str(pair.get("child", "")).strip()
            if parent and child:
                pairs.add((parent, child))
    return pairs


def _positive_examples_from_docs(
    docs: list[dict],
    max_examples: int | None,
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for doc in docs:
        for pair in doc.get("taxonomic_pairs", []) or []:
            parent = str(pair.get("parent", "")).strip()
            child = str(pair.get("child", "")).strip()
            key = (parent, child)
            if not parent or not child or key in seen:
                continue
            seen.add(key)
            examples.append({"parent": parent, "child": child, "is_parent": True})
            if max_examples is not None and len(examples) >= max_examples:
                return examples
    return examples


def _negative_examples_from_positive_pairs(
    positive_examples: list[dict[str, object]],
    gold_pairs: set[tuple[str, str]],
    max_examples: int,
    seed: int,
) -> list[dict[str, object]]:
    if max_examples <= 0 or len(positive_examples) < 2:
        return []

    parents = list(dict.fromkeys(str(e["parent"]) for e in positive_examples))
    children = list(dict.fromkeys(str(e["child"]) for e in positive_examples))
    positive_pairs = {
        (str(e["parent"]), str(e["child"]))
        for e in positive_examples
    }

    candidates: list[tuple[str, str]] = []
    for parent in parents:
        for child in children:
            key = (parent, child)
            if parent == child or key in positive_pairs or key in gold_pairs:
                continue
            candidates.append(key)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    return [
        {"parent": parent, "child": child, "is_parent": False}
        for parent, child in candidates[:max_examples]
    ]


def _stable_seed(seed: int | None, doc_id: str) -> int:
    base = 0 if seed is None else seed
    digest = hashlib.sha256(f"{base}:{doc_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _prepare_doc_candidates(
    docs_with_pairs: dict[str, list[str]],
    client: LLMClient,
    cluster_types: bool,
    n_clusters: int,
    workers: int,
    max_consecutive_errors: int,
) -> list[_DocCandidates]:
    if not cluster_types:
        return [
            _DocCandidates(
                doc_id=doc_id,
                type_vocab=type_vocab,
                candidates=_generate_candidates(type_vocab),
            )
            for doc_id, type_vocab in docs_with_pairs.items()
        ]

    prepared: list[_DocCandidates] = []
    errors = 0
    consecutive_errors = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_doc = {
            executor.submit(
                _prepare_single_doc_candidates, doc_id, type_vocab, client, n_clusters,
            ): doc_id
            for doc_id, type_vocab in docs_with_pairs.items()
        }
        for future in progress_bar(
            as_completed(future_to_doc),
            total=len(future_to_doc),
            desc="S4 candidate docs",
            unit="doc",
        ):
            doc_id = future_to_doc[future]
            try:
                prepared.append(future.result())
                consecutive_errors = 0
            except (LLMError, Exception):
                errors += 1
                consecutive_errors += 1
                logger.error(
                    "S4 doc %s: candidate preparation failed (errors so far: %d)",
                    doc_id, errors, exc_info=True,
                )
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"S4 aborted: {consecutive_errors} consecutive candidate-prep failures"
                    )

    return prepared


def _prepare_single_doc_candidates(
    doc_id: str,
    type_vocab: list[str],
    client: LLMClient,
    n_clusters: int,
) -> _DocCandidates:
    if len(type_vocab) > n_clusters:
        try:
            clusters = _cluster_types(
                type_vocab,
                n_clusters,
                client,
                call_label=f"clustering doc={doc_id}",
            )
            candidates = _generate_candidates_clustered(clusters)
        except (LLMError, Exception):
            logger.warning("S4 doc %s: clustering failed, using full pairwise", doc_id)
            candidates = _generate_candidates(type_vocab)
    else:
        candidates = _generate_candidates(type_vocab)

    return _DocCandidates(doc_id=doc_id, type_vocab=type_vocab, candidates=candidates)


def _make_batch_tasks(
    doc_candidates: list[_DocCandidates],
    batch_size: int,
    examples_by_doc: dict[str, list[dict[str, object]]],
    *,
    mode: str = "labels",
) -> list[_BatchTask]:
    tasks: list[_BatchTask] = []
    for doc in doc_candidates:
        if mode in _CHILD_GROUPED_BATCH_MODES:
            batches = _group_candidates_by_child(
                doc.candidates,
                batch_size,
                split_oversized=mode not in _SINGLE_PARENT_PROMPT_MODES,
            )
        else:
            batches = [
                doc.candidates[start:start + batch_size]
                for start in range(0, len(doc.candidates), batch_size)
            ]

        for batch_index, pairs in enumerate(batches):
            tasks.append(_BatchTask(
                doc_id=doc.doc_id,
                batch_index=batch_index,
                pairs=pairs,
                examples=examples_by_doc.get(doc.doc_id, []),
            ))
    return tasks


def _group_candidates_by_child(
    candidates: list[tuple[str, str]],
    batch_size: int,
    *,
    split_oversized: bool = True,
) -> list[list[tuple[str, str]]]:
    candidates_by_child: dict[str, list[tuple[str, str]]] = {}
    for pair in candidates:
        candidates_by_child.setdefault(pair[1], []).append(pair)

    batches: list[list[tuple[str, str]]] = []
    current_batch: list[tuple[str, str]] = []
    for child_pairs in candidates_by_child.values():
        if len(child_pairs) > batch_size:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
            if split_oversized:
                batches.extend(
                    child_pairs[start:start + batch_size]
                    for start in range(0, len(child_pairs), batch_size)
                )
            else:
                batches.append(child_pairs)
            continue

        if current_batch and len(current_batch) + len(child_pairs) > batch_size:
            batches.append(current_batch)
            current_batch = []
        current_batch.extend(child_pairs)

    if current_batch:
        batches.append(current_batch)
    return batches


def _dedupe_doc_results(
    doc_id: str,
    confirmed: list[TaxonomyResult],
) -> list[TaxonomyResult]:
    seen: set[tuple[str, str]] = set()
    unique: list[TaxonomyResult] = []
    for r in confirmed:
        key = (r.parent, r.child)
        if key not in seen:
            seen.add(key)
            unique.append(TaxonomyResult(parent=r.parent, child=r.child, source_doc_ids=[doc_id]))
    return unique


def _run_depth_passes(
    doc_candidates: list[_DocCandidates],
    results_by_doc: dict[str, list[TaxonomyResult]],
    client: LLMClient,
    workers: int,
) -> None:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_doc = {
            executor.submit(
                _run_depth_pass_for_doc, doc, results_by_doc.get(doc.doc_id, []), client,
            ): doc
            for doc in doc_candidates
            if len(doc.type_vocab) > 1
        }
        for future in progress_bar(
            as_completed(future_to_doc),
            total=len(future_to_doc),
            desc="S4 depth docs",
            unit="doc",
        ):
            doc = future_to_doc[future]
            try:
                results_by_doc[doc.doc_id] = future.result()
            except (LLMError, Exception):
                logger.warning(
                    "S4 doc %s: depth pass failed, keeping pass-1 results",
                    doc.doc_id, exc_info=True,
                )


def _run_depth_pass_for_doc(
    doc: _DocCandidates,
    unique: list[TaxonomyResult],
    client: LLMClient,
) -> list[TaxonomyResult]:
    seen = {(r.parent, r.child) for r in unique}
    result = list(unique)
    for r in _depth_pass(doc.type_vocab, unique, client):
        key = (r.parent, r.child)
        if key not in seen:
            seen.add(key)
            result.append(
                TaxonomyResult(parent=r.parent, child=r.child, source_doc_ids=[doc.doc_id])
            )
    return result


def _generate_candidates(type_vocab: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for t1, t2 in combinations(type_vocab, 2):
        pairs.append((t1, t2))
        pairs.append((t2, t1))
    return list(dict.fromkeys(pairs))


def _cluster_types(
    type_vocab: list[str],
    n_clusters: int,
    client: LLMClient,
    *,
    call_label: str = "clustering",
) -> dict[str, list[str]]:
    messages = build_s4_cluster_messages(type_vocab, n_clusters)
    response = _call_s4_json(messages, client, call_label)

    vocab_set = set(type_vocab)
    assigned: set[str] = set()
    result: dict[str, list[str]] = {}
    for cid, members in response.get("clusters", {}).items():
        valid = [m for m in members if m in vocab_set]
        if valid:
            result[str(cid)] = valid
            assigned.update(valid)

    unassigned = [t for t in type_vocab if t not in assigned]
    if unassigned:
        logger.warning(
            "S4 clustering: %d unassigned types added to catch-all cluster", len(unassigned)
        )
        result[str(len(result))] = unassigned

    return result


def _generate_candidates_clustered(
    clusters: dict[str, list[str]],
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for members in clusters.values():
        for t1, t2 in combinations(sorted(members), 2):
            pairs.append((t1, t2))
            pairs.append((t2, t1))
    return list(dict.fromkeys(pairs))


def _debug_enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "y", "on"}


def _print_s4_prompt(call_label: str, messages: list[dict[str, str]]) -> None:
    with _DEBUG_PRINT_LOCK:
        print(f"\n===== S4 PROMPT START {call_label} =====", file=sys.stderr)
        for index, message in enumerate(messages):
            print(
                f"\n--- message[{index}] role={message.get('role', '')} ---",
                file=sys.stderr,
            )
            print(message.get("content", ""), file=sys.stderr)
        print(
            f"===== S4 PROMPT END {call_label} =====\n",
            file=sys.stderr,
            flush=True,
        )


def _print_s4_raw_output(call_label: str, text: str) -> None:
    with _DEBUG_PRINT_LOCK:
        print(f"\n===== S4 RAW OUTPUT START {call_label} =====", file=sys.stderr)
        print(text, file=sys.stderr)
        print(
            f"===== S4 RAW OUTPUT END {call_label} =====\n",
            file=sys.stderr,
            flush=True,
        )


def _call_s4_json(
    messages: list[dict[str, str]],
    client: LLMClient,
    call_label: str,
) -> dict:
    print_prompt = _debug_enabled("PIPELINE_PRINT_S4_PROMPTS")
    print_raw = _debug_enabled("PIPELINE_PRINT_S4_RAW_OUTPUTS")
    if print_prompt:
        _print_s4_prompt(call_label, messages)
    if not print_raw:
        return client.chat_json(messages)

    text = client.chat(messages)
    _print_s4_raw_output(call_label, text)
    return _extract_json(text)


def _call_batch(
    pairs: list[tuple[str, str]],
    client: LLMClient,
    *,
    mode: str = "labels",
    examples: list[dict[str, object]] | None = None,
    call_label: str = "classification",
) -> list[TaxonomyResult]:
    prompt_mode = "boolean" if mode == "boolean" else "labels"
    example_style = "grouped" if mode in _GROUPED_FEWSHOT_MODES else "sequential"
    messages = build_s4_messages(
        pairs,
        mode=prompt_mode,
        examples=examples,
        example_style=example_style,
        single_parent_per_child=mode in _SINGLE_PARENT_PROMPT_MODES,
    )
    response = _call_s4_json(messages, client, call_label)

    confirmed = []
    for entry in response.get("results", []):
        is_parent = _entry_is_parent(entry, mode)
        if not is_parent:
            continue

        try:
            pair_index = int(entry.get("pair_index"))
        except (TypeError, ValueError):
            logger.warning("S4: TAXONOMIC_IS_A entry missing valid pair_index: %r", entry)
            continue
        if pair_index < 0 or pair_index >= len(pairs):
            logger.warning("S4: TAXONOMIC_IS_A entry has out-of-range pair_index: %r", entry)
            continue

        parent, child = pairs[pair_index]
        if parent and child:
            confirmed.append(TaxonomyResult(parent=parent, child=child))
    return confirmed


def _entry_is_parent(entry: dict, mode: str) -> bool:
    if mode == "boolean":
        raw = entry.get("is_parent")
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            value = raw.strip().lower()
            if value in {"true", "1", "yes", "y"}:
                return True
            if value in {"false", "0", "no", "n"}:
                return False

    relation_label = str(entry.get("relation_label", "")).strip().upper()
    return relation_label == "TAXONOMIC_IS_A"


def _depth_pass(
    type_vocab: list[str],
    confirmed: list[TaxonomyResult],
    client: LLMClient,
) -> list[TaxonomyResult]:
    known: dict[str, set[str]] = {t: set() for t in type_vocab}
    for r in confirmed:
        if r.parent in known:
            known[r.parent].add(r.child)

    additions: list[TaxonomyResult] = []
    for parent, children in known.items():
        try:
            for child in _call_depth_single(parent, sorted(children), type_vocab, client):
                if child in known and child != parent:
                    additions.append(TaxonomyResult(parent=parent, child=child))
        except (LLMError, Exception):
            logger.warning("S4 depth pass failed for parent %r", parent, exc_info=True)
    return additions


def _call_depth_single(
    parent: str,
    known_children: list[str],
    all_types: list[str],
    client: LLMClient,
) -> list[str]:
    candidates = [t for t in all_types if t != parent and t not in set(known_children)]
    if not candidates:
        return []
    messages = build_s4_depth_messages(parent, known_children, candidates)
    response = _call_s4_json(messages, client, f"depth parent={parent!r}")
    return [t.strip() for t in response.get("additional_subtypes", []) if t.strip()]


def _token_overlap_f1(a: str, b: str) -> float:
    """Word-level token overlap F1 between two type label strings.

    Optional post-filter: a (child, parent) pair where
    the labels share few tokens is unlikely to be a genuine is-a relation.
    Example: "transmission type value automatic" vs "transmission type value"
    → F1 ≈ 0.86 (high, correct). "temperature sensor" vs "alkali metal"
    → F1 = 0.0 (no overlap, spurious match).
    """
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    intersection = len(ta & tb)
    return 2 * intersection / (len(ta) + len(tb))


def run_embedding(
    term_typings: list[TermTypingResult],
    taxonomy_parent_index: TaxonomyParentIndex,
    k: int = 1,
    threshold: float = 0.0,
    token_overlap_threshold: float = 0.0,
) -> list[TaxonomyResult]:
    """Discover is-a relationships via embedding similarity, per document.

    For each document, finds the k nearest training parent types for each of
    its types by cosine similarity, optionally filtered by token overlap F1


    Args:
        token_overlap_threshold: Minimum word-level token overlap F1 between
            child and parent labels. Filters out spurious embedding matches
            where the labels share no common tokens (e.g. "sensor" → "metal").
            Set to 0.0 to disable (default). Typical useful range: 0.3–0.5.
    """
    # Group types by document
    doc_to_types: dict[str, set[str]] = defaultdict(set)
    for tt in term_typings:
        for doc_id in tt.source_doc_ids:
            doc_to_types[doc_id].update(tt.types)

    result: list[TaxonomyResult] = []
    for doc_id, types in doc_to_types.items():
        type_vocab = sorted(types)
        if not type_vocab:
            continue

        results_by_child = taxonomy_parent_index.batch_query(type_vocab, k=k)
        seen: set[tuple[str, str]] = set()
        for child, parent_scores in zip(type_vocab, results_by_child):
            for parent, score in parent_scores:
                if score >= threshold and parent != child:
                    if token_overlap_threshold > 0.0 and \
                            _token_overlap_f1(child, parent) < token_overlap_threshold:
                        continue
                    key = (parent, child)
                    if key not in seen:
                        seen.add(key)
                        result.append(TaxonomyResult(
                            parent=parent, child=child, source_doc_ids=[doc_id],
                        ))

    logger.info(
        "S4 embedding: %d is-a relations emitted across %d documents (threshold=%.2f)",
        len(result), len(doc_to_types), threshold,
    )
    return result
