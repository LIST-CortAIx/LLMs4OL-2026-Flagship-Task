"""Step 3 - Term Typing (RAG-based).

Free generation grounded by the full document, using the k nearest gold
(term, type) pairs as few-shot, then vocabulary snapping to the S2 types.
Per-document processing: each document's terms are typed in one LLM call, with
the document text supplied once as grounding context.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from ontology_learning.llm import LLMClient, LLMError
from ontology_learning.models import ExtractedTerm, TermTypingResult
from ontology_learning.prompts import build_s3_rag_messages, build_s3_constrained_messages
from ontology_learning.steps.s0_retriever import Retriever

logger = logging.getLogger(__name__)


# ── Context helpers (full-doc / sentence / title grounding) ─────────────────

def _doc_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("title:"):
            return line[6:].strip()
    return ""


def _containing_sentence(text: str, term: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    tl = term.lower()
    for s in sentences:
        if tl in s.lower():
            return s.strip()
    return ""


def _build_contexts(terms, doc_text, mode):
    """Return (per_term_contexts | None, doc_context | None, context_chars)."""
    if mode == "none" or not doc_text:
        return None, None, 150
    if mode == "full":
        return None, doc_text, 150
    title = _doc_title(doc_text)
    ctxs = []
    for t in terms:
        sent = _containing_sentence(doc_text, t.text)
        if mode == "title_sentence":
            parts = ([f"Title: {title}"] if title else []) + ([sent] if sent else [])
            ctxs.append(". ".join(parts))
        else:  # "sentence"
            ctxs.append(sent)
    return ctxs, None, 400


def run(
    terms: list[ExtractedTerm],
    client: LLMClient,
    retriever: Retriever,
    k: int = 10,
    workers: int = 4,
    max_consecutive_errors: int = 5,
    type_vocab: set[str] | None = None,
    documents: list[dict] | None = None,
    context_mode: str = "full",
    batch_size: int = 0,
    constrained: bool = False,
    lenient: bool = False,
    augment_frequent: int = 0,
    hybrid: bool = False,
    hybrid_alpha: float = 0.5,
) -> list[TermTypingResult]:
    """Assign semantic types to every extracted term via RAG few-shot prompting.

    Args:
        terms: Output of S1 — extracted terms with optional context sentences.
        client: Configured LLM client.
        retriever: Embedding retriever (gold term→type pairs) from the S0 index.
        k: Nearest-neighbour examples retrieved per term (best: 10).
        workers: Maximum parallel LLM calls.
        type_vocab: Optional S2 type set — out-of-vocab predictions are snapped to
            the nearest S2 type by cosine (keeps the ontology internally consistent).
        documents: ``[{"id", "text"}]`` corpus — required for context grounding.
        context_mode: ``full`` (whole document, best) | ``sentence`` | ``title_sentence`` | ``none``.
        batch_size: Terms per LLM call within a document; ``0`` = the whole document
            in one call (best — sends the document text once).

    Returns:
        One :class:`TermTypingResult` per input term, preserving input order.
    """
    doc_text = {str(d["id"]): d.get("text", "") for d in documents} if documents else {}
    use_ctx = bool(doc_text) and context_mode != "none"

    # Group terms by their primary (first) source document so each term is typed
    # once, with that document's text for grounding.
    batches: list[tuple[str, list[ExtractedTerm]]] = []
    if use_ctx:
        by_doc: dict[str, list[ExtractedTerm]] = {}
        for t in terms:
            did = str(t.source_doc_ids[0]) if t.source_doc_ids else ""
            by_doc.setdefault(did, []).append(t)
        for did, dterms in by_doc.items():
            bs = len(dterms) if batch_size <= 0 else batch_size
            for i in range(0, len(dterms), max(bs, 1)):
                batches.append((did, dterms[i:i + max(bs, 1)]))
    else:
        bs = batch_size if batch_size > 0 else 10
        for i in range(0, len(terms), bs):
            batches.append(("", terms[i:i + bs]))

    logger.info("S3: %d terms → %d batches (k=%d, context=%s, workers=%d)",
                len(terms), len(batches), k, context_mode if use_ctx else "none", workers)

    results: dict[str, list[str]] = {}
    errors = 0
    consecutive_errors = 0

    retriever.warm()

    # Optional frequent-type prior pool (built once from the index)
    augment_types: list[str] = []
    if augment_frequent > 0:
        from collections import Counter
        c = Counter()
        for p in retriever.pairs:
            for ty in p.get("types", []):
                t = ty.strip().lower()
                if t:
                    c[t] += 1
        augment_types = [t for t, _ in c.most_common(augment_frequent)]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        fut = {executor.submit(_call_batch, bterms, retriever, k, client,
                               doc_text.get(did, ""), context_mode if use_ctx else "none",
                               constrained, lenient, augment_types, hybrid, hybrid_alpha): bterms
               for did, bterms in batches}
        for future in as_completed(fut):
            bterms = fut[future]
            try:
                results.update(future.result())
                consecutive_errors = 0
            except (LLMError, Exception):
                errors += 1
                consecutive_errors += 1
                logger.error("S3: batch failed for terms %s (errors so far: %d)",
                             [t.text for t in bterms[:3]], errors, exc_info=True)
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(f"S3 aborted: {consecutive_errors} consecutive LLM failures")

    typed = [
        TermTypingResult(term=t.text, types=results.get(t.text, []), source_doc_ids=t.source_doc_ids)
        for t in terms
    ]
    if type_vocab:
        typed = _snap_to_vocab(typed, type_vocab, retriever)
    assigned = sum(1 for r in typed if r.types)
    logger.info("S3 done: %d/%d terms typed, %d batches failed", assigned, len(typed), errors)
    return typed


def _snap_to_vocab(
    typed: list[TermTypingResult],
    type_vocab: set[str],
    retriever: Retriever,
) -> list[TermTypingResult]:
    """Snap any out-of-vocabulary type to the nearest S2 type by cosine similarity."""
    import numpy as np

    oov = {t for r in typed for t in r.types if t not in type_vocab}
    if not oov:
        return typed

    if retriever._model is None:
        from sentence_transformers import SentenceTransformer
        retriever._model = SentenceTransformer(retriever.model_name)

    vocab_list = sorted(type_vocab)
    vocab_emb = retriever._model.encode(
        vocab_list, normalize_embeddings=True, show_progress_bar=False
    )
    oov_list = sorted(oov)
    oov_emb = retriever._model.encode(
        oov_list, normalize_embeddings=True, show_progress_bar=False
    )
    scores = oov_emb @ vocab_emb.T  # (|oov|, |vocab|)
    snap_map = {oov_list[i]: vocab_list[int(scores[i].argmax())] for i in range(len(oov_list))}

    logger.info(
        "S3 vocab snap: %d out-of-vocabulary type(s) remapped: %s",
        len(snap_map),
        {k: v for k, v in snap_map.items()},
    )

    return [
        TermTypingResult(
            term=r.term,
            types=list(dict.fromkeys(snap_map.get(t, t) for t in r.types)),
            source_doc_ids=r.source_doc_ids,
        )
        for r in typed
    ]


def _call_batch(
    terms: list[ExtractedTerm],
    retriever: Retriever,
    k: int,
    client: LLMClient,
    doc_text: str = "",
    context_mode: str = "none",
    constrained: bool = False,
    lenient: bool = False,
    augment_types: list[str] = (),
    hybrid: bool = False,
    hybrid_alpha: float = 0.5,
) -> dict[str, list[str]]:
    """Run one LLM call for a batch (one document's terms), returning term → types."""
    contexts, doc_context, context_chars = _build_contexts(terms, doc_text, context_mode)
    if context_mode != "none":
        # Context path (matches debug_s3): query = term, or term+sentence for sentence modes.
        queries = [
            f"{t.text}: {contexts[i][:100]}" if contexts and contexts[i] else t.text
            for i, t in enumerate(terms)
        ]
    else:
        # Legacy path: query uses the term's own context sentence from S1.
        queries = [
            f"{t.text}: {t.context_sentence[:100]}" if t.context_sentence else t.text
            for t in terms
        ]
    term_examples = retriever.batch_query(queries, k=k, hybrid=hybrid, alpha=hybrid_alpha,
                                          lex_keys=[t.text for t in terms])

    if constrained or lenient:
        candidates = []
        for exs in term_examples:
            seen, cands = set(), []
            for ex in exs:
                for ty in ex.get("types", []):
                    t = ty.strip().lower()
                    if t and t not in seen:
                        seen.add(t); cands.append(t)
            candidates.append(cands)
        messages = build_s3_constrained_messages(
            terms, candidates, contexts=contexts, doc_context=doc_context,
            context_chars=context_chars, lenient=lenient, frequent_types=augment_types)
    else:
        messages = build_s3_rag_messages(terms, term_examples, contexts=contexts,
                                         doc_context=doc_context, context_chars=context_chars,
                                         frequent_types=augment_types)
    response = client.chat_json(messages)

    out: dict[str, list[str]] = {}
    for entry in response.get("results", []):
        term_text = entry.get("term", "").strip()
        if not term_text:
            idx = int(entry.get("term_index", -1))
            if 0 <= idx < len(terms):
                term_text = terms[idx].text
        if term_text:
            raw_types = [t.strip().lower() for t in entry.get("types", []) if t.strip()]
            out[term_text] = raw_types
    return out
