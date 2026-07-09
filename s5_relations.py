"""Step 5 — Non-Taxonomic Relation Extraction.

For each document, extracts non-taxonomic semantic relations using two sources
combined in a single LLM call:

1. TEXT EXTRACTION — relations explicitly stated or implied in the document text.
2. SEMANTIC INFERENCE — structural ontology axioms between the document's types
   inferred from domain knowledge (e.g. disjoint with, equivalent class), even
   when not mentioned in the text.

Per-document scale makes this natural: 2–5 types per document means all type
pairs fit in one prompt — no clustering stage needed.

Domain inference pre-step: one LLM call infers domain_name and domain_description
from the global type vocabulary and sample passages, injected into every
per-document prompt.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def gate_relation_documents(documents: list[dict], classifier_path: str | Path,
                            threshold: float | None = None) -> list[dict]:
    """Keep only documents predicted to HAVE a non-taxonomic relation (S1 R4 analog).

    ~93% of docs have no S5 relation; this TF-IDF classifier filters them out before
    the LLM so the extraction prompt stays clean for relation-bearing docs. A higher
    ``threshold`` filters more docs (macro-F1 favours aggressive filtering of the
    dominant empty class); ``None`` uses the classifier's trained threshold.
    """
    import pickle
    blob = pickle.loads(Path(classifier_path).read_bytes())
    pipeline = blob["pipeline"]
    threshold = blob["threshold"] if threshold is None else threshold
    proba = pipeline.predict_proba([d.get("text", "") for d in documents])[:, 1]
    kept = [d for d, p in zip(documents, proba) if p >= threshold]
    logger.info("S5 relation gate: %d/%d docs kept (%.1f%% filtered as relation-free)",
                len(kept), len(documents), 100 * (1 - len(kept) / max(len(documents), 1)))
    return kept


def snap_doc_triples(
    triples: list[tuple[str, str, str]],
    doc_types: list[str],
    model: Any,
) -> list[tuple[str, str, str]]:
    """Snap each triple's head/tail to the document's S2 type vocabulary (S3-style).

    In-vocab entities (normalized match) take the canonical S2 spelling; out-of-vocab
    entities are snapped to the nearest S2 type by embedding cosine. S5 relations are
    type–type and 100% of gold entities are S2 types, so this recovers paraphrase
    near-misses without changing the relation label.
    """
    if not doc_types or not triples or model is None:
        return triples
    tnorm = {_norm(t): t for t in doc_types}
    oov = sorted({e for s, _, o in triples for e in (s, o) if _norm(e) not in tnorm})
    snap: dict[str, str] = {}
    if oov:
        import numpy as np
        vocab = sorted(set(doc_types))
        ve = model.encode(vocab, normalize_embeddings=True, show_progress_bar=False)
        oe = model.encode(oov, normalize_embeddings=True, show_progress_bar=False)
        sims = oe @ ve.T
        for i, e in enumerate(oov):
            snap[e] = vocab[int(sims[i].argmax())]

    def m(e: str) -> str:
        return tnorm.get(_norm(e)) or snap.get(e, e)

    return [(m(s), r, m(o)) for s, r, o in triples]

from llm import LLMClient, LLMError
from models import ExtractedTerm, ExtractedType, RelationResult, TermTypingResult
from prompts import build_s5_domain_messages, build_s5_messages

logger = logging.getLogger(__name__)


def load_example_pool(
    gold_path: str | Path,
    s2_gold_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Map training doc id → ``{"types": [...], "triples": [...]}`` for few-shot.

    Only docs that actually have non-taxonomic relations are included (the retriever
    pool excludes the ~94% relation-free training docs). ``types`` is the doc's full
    S2 vocabulary (from ``s2_gold_path`` if given, else the entities in its triples)
    so each example is a complete ``types → relations`` demonstration.
    """
    data = json.loads(Path(gold_path).read_text(encoding="utf-8"))["non_taxonomic_relations"]
    triples: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in data:
        doc_ids = r.get("source_doc_ids") or ([r["source_doc_id"]] if r.get("source_doc_id") else [])
        for d in doc_ids:
            triples[str(d)].append({"subject": r["head"], "relation": r["relation"], "object": r["tail"]})

    types: dict[str, set[str]] = defaultdict(set)
    if s2_gold_path and Path(s2_gold_path).exists():
        for t in json.loads(Path(s2_gold_path).read_text(encoding="utf-8"))["types"]:
            for d in t.get("source_doc_ids", []):
                types[str(d)].add(t["text"])

    pool: dict[str, dict[str, Any]] = {}
    for d, trs in triples.items():
        tset = types.get(d) or {e for tr in trs for e in (tr["subject"], tr["object"])}
        pool[d] = {"types": sorted(tset), "triples": trs}
    return pool


def load_full_example_pool(
    s2_gold_path: str | Path,
    s5_gold_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Map EVERY training doc id → ``{"types", "triples"}`` (triples ``[]`` for the
    ~93% relation-free docs).

    Unlike ``load_example_pool`` (relation-bearing only), this lets few-shot retrieval
    be NEIGHBOURHOOD-FAITHFUL: a query is shown its k nearest training docs with their
    TRUE relations — empty demonstrations for relation-free neighbours — so the model
    learns restraint from realistic examples that match the query's neighbourhood,
    conveying prevalence through examples rather than a prompt claim.
    """
    types: dict[str, set[str]] = defaultdict(set)
    for t in json.loads(Path(s2_gold_path).read_text(encoding="utf-8"))["types"]:
        for d in t.get("source_doc_ids", []):
            types[str(d)].add(t["text"])
    triples: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in json.loads(Path(s5_gold_path).read_text(encoding="utf-8"))["non_taxonomic_relations"]:
        doc_ids = r.get("source_doc_ids") or ([r["source_doc_id"]] if r.get("source_doc_id") else [])
        for d in doc_ids:
            triples[str(d)].append({"subject": r["head"], "relation": r["relation"], "object": r["tail"]})
    return {d: {"types": sorted(tset), "triples": triples.get(d, [])} for d, tset in types.items()}


def _cover_relations(pos, neighbours, example_pool, exclude_id, with_text,
                     diverse_relations, make_item) -> None:
    """Diversity-aware coverage: append the nearest pool doc featuring each requested
    relation that the nearest-neighbour positives don't already cover.

    Lifts recall of under-shown text-grounded relations (has part / part_of / located in
    / database_cross_reference / …) without removing the similarity-ranked positives.
    Mutates ``pos`` in place. Recall-only — adds positive demos, never negatives.
    """
    if not diverse_relations:
        return
    want = {_norm(r) for r in diverse_relations}
    have = {_norm(t["relation"]) for it in pos for t in it["triples"]}
    for nd in neighbours:
        if want <= have:
            break
        tid = str(nd.get("id"))
        if tid == str(exclude_id) or tid not in example_pool:
            continue
        rels = {_norm(t["relation"]) for t in example_pool[tid]["triples"]}
        if (want - have) & rels:
            item = make_item(tid, nd)
            if item not in pos:
                pos.append(item)
                have |= rels


def retrieve_examples(
    retriever: Any,
    example_pool: dict[str, dict[str, Any]],
    query: str,
    k: int,
    exclude_id: str | None = None,
    with_text: bool = False,
    by_types: bool = False,
    n_neg: int = 0,
    min_pos: int | None = None,
    diverse_relations: list[str] | None = None,
) -> list[dict[str, Any]] | None:
    """Few-shot demonstrations for one doc: the ``k`` nearest training docs *that
    have relations*, each as ``{"types", "triples"[, "text"]}`` — a full
    ``types → relations`` mapping. Searches a wide neighbourhood since only ~9% of
    train docs have relations (query cost is O(N) regardless of neighbourhood).

    ``by_types``: rank by **type-vocabulary** similarity (query = the doc's S2 types,
    matched against training docs' type embeddings) instead of doc-text similarity.
    ``n_neg``: also include the nearest ``n_neg`` *relation-free* docs as negative
    demos (``types → (none)``) to teach restraint / curb over-extraction.

    ``min_pos`` (neighbourhood-faithful, requires a FULL pool incl. relation-free docs):
    take the ``min_pos`` nearest relation-bearing demos + the ``k - min_pos`` nearest
    relation-free demos (shown ``types → (none)``). The positive floor protects the
    relation-bearing docs' recall while the empty demos drive restraint; the ratio is
    the dial between empty-acc and the 61's F1. ``None`` = legacy behaviour.
    """
    if not retriever or not example_pool:
        return None
    n = max((k + n_neg) * 100, 500)
    neighbours = (retriever.query_by_type_labels(query, k=n) if by_types
                  else retriever.query(query, k=n))

    def _item(tid: str, nd: dict) -> dict[str, Any]:
        item: dict[str, Any] = {"types": example_pool[tid]["types"],
                                "triples": example_pool[tid]["triples"]}
        if with_text and nd.get("text"):
            item["text"] = nd["text"]
        return item

    if min_pos is None:
        # Legacy: k nearest in-pool demos as positives + n_neg out-of-pool negatives.
        pos: list[dict[str, Any]] = []
        neg: list[dict[str, Any]] = []
        for nd in neighbours:
            tid = str(nd.get("id"))
            if tid == str(exclude_id):
                continue
            if tid in example_pool and len(pos) < k:
                pos.append(_item(tid, nd))
            elif n_neg and tid not in example_pool and nd.get("types") and len(neg) < n_neg:
                item = {"types": list(nd["types"]), "triples": []}
                if with_text and nd.get("text"):
                    item["text"] = nd["text"]
                neg.append(item)
            if len(pos) >= k and len(neg) >= n_neg:
                break
        _cover_relations(pos, neighbours, example_pool, exclude_id, with_text,
                         diverse_relations, _item)
        return (pos + neg) or None

    # Neighbourhood-faithful with a positive floor (needs a full pool).
    n_neg_slots = max(k - min_pos, 0)
    pos, neg = [], []
    for nd in neighbours:
        tid = str(nd.get("id"))
        if tid == str(exclude_id) or tid not in example_pool:
            continue
        if example_pool[tid]["triples"]:
            if len(pos) < min_pos:
                pos.append(_item(tid, nd))
        elif len(neg) < n_neg_slots:
            neg.append(_item(tid, nd))
        if len(pos) >= min_pos and len(neg) >= n_neg_slots:
            break
    _cover_relations(pos, neighbours, example_pool, exclude_id, with_text,
                     diverse_relations, _item)
    return (pos + neg) or None


def run(
    documents: list[dict[str, str]],
    terms: list[ExtractedTerm],
    term_typings: list[TermTypingResult],
    client: LLMClient,
    types: list[ExtractedType] | None = None,
    domain_inference: bool = True,
    workers: int = 4,
    max_consecutive_errors: int = 5,
    examples: list[dict[str, Any]] | None = None,
    retriever: Any = None,
    example_pool: dict[str, dict[str, Any]] | None = None,
    k_examples: int = 0,
    examples_with_text: bool = False,
    examples_by_types: bool = False,
    n_neg_examples: int = 0,
    snap: bool = False,
    relation_classifier_path: str | None = None,
    relation_gate_threshold: float | None = None,
    prompt_v2: bool = False,
) -> list[RelationResult]:
    """Extract non-taxonomic relations from document passages (open RE).

    Args:
        documents: Original documents with ``id`` and ``text`` fields.
        terms: S1 output carrying term → source_doc_ids links.
        term_typings: S3 output (term → types mappings).
        client: Configured LLM client.
        types: S2 output (type vocabulary). When provided, used for domain
            inference; otherwise types are derived from term_typings.
        domain_inference: When True, one LLM call infers domain_name and
            domain_description from the type vocabulary and sample passages;
            the description is injected into every extraction prompt (P1).
        workers: Maximum parallel LLM calls.
        max_consecutive_errors: Abort if this many calls fail in a row.
        examples: Optional few-shot examples (``{"subject", "relation", "object"}``).

    Returns:
        Deduplicated list of ``RelationResult`` triples, each with
        ``source_doc_id`` set to the originating document.

    Raises:
        RuntimeError: If ``max_consecutive_errors`` is reached.
    """
    # Empty-doc gate: drop docs predicted to have no non-taxonomic relation (S1 R4 analog).
    if relation_classifier_path:
        documents = gate_relation_documents(documents, relation_classifier_path,
                                            threshold=relation_gate_threshold)
        if not documents:
            logger.info("S5: all documents filtered as relation-free → no relations")
            return []

    type_labels: list[str]
    if types is not None:
        type_labels = [t.text for t in types]
    else:
        type_labels = sorted({t for tt in term_typings for t in tt.types})

    typing_map: dict[str, list[str]] = {tt.term: tt.types for tt in term_typings}
    term_to_docs: dict[str, list[str]] = {t.text: t.source_doc_ids for t in terms}

    doc_typed_terms: dict[str, list[tuple[str, str]]] = {d["id"]: [] for d in documents}
    for term_text, doc_ids in term_to_docs.items():
        for typ in typing_map.get(term_text, []):
            for doc_id in doc_ids:
                if doc_id in doc_typed_terms:
                    doc_typed_terms[doc_id].append((term_text, typ))

    # Full per-document S2 type vocabulary (S5 relations hold between types, which
    # may not all have a typed term). Passed explicitly so the model sees the whole
    # vocabulary for semantic inference.
    doc_types: dict[str, list[str]] = {d["id"]: [] for d in documents}
    if types is not None:
        for ty in types:
            for doc_id in ty.source_doc_ids:
                if doc_id in doc_types:
                    doc_types[doc_id].append(ty.text)

    domain_description: str | None = None
    if domain_inference and type_labels:
        sample_passages = [d["text"] for d in documents[:5]]
        try:
            domain_description = _infer_domain(type_labels, sample_passages, client)
            logger.debug("S5 domain inference: %r", domain_description)
        except (LLMError, Exception):
            logger.warning(
                "S5 domain inference failed, continuing without it", exc_info=True
            )

    logger.info(
        "S5: %d documents, domain_inference=%s", len(documents), domain_inference
    )

    relations: list[RelationResult] = []
    errors = 0
    consecutive_errors = 0

    use_fewshot = bool(retriever and example_pool and k_examples > 0)
    if use_fewshot:
        logger.info("S5 few-shot: k=%d from a pool of %d training docs",
                    k_examples, len(example_pool))

    def _examples_for(doc: dict[str, str]) -> list[dict[str, Any]] | None:
        if use_fewshot:
            if examples_by_types:
                query = ", ".join(sorted(set(doc_types.get(doc["id"], [])))) or doc.get("text", "")
            else:
                query = doc.get("text", "")
            return retrieve_examples(retriever, example_pool, query,
                                     k_examples, exclude_id=doc["id"],
                                     with_text=examples_with_text, by_types=examples_by_types,
                                     n_neg=n_neg_examples)
        return examples

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_doc = {
            executor.submit(
                _call_document,
                doc,
                doc_typed_terms.get(doc["id"], []),
                domain_description,
                client,
                _examples_for(doc),
                sorted(set(doc_types.get(doc["id"], []))) or None,
                prompt_v2,
            ): doc
            for doc in documents
        }
        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            try:
                doc_relations = future.result()
                consecutive_errors = 0
                relations.extend(doc_relations)
            except (LLMError, Exception):
                errors += 1
                consecutive_errors += 1
                logger.error(
                    "S5: extraction failed for doc %r (errors so far: %d)",
                    doc["id"], errors, exc_info=True,
                )
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"S5 aborted: {consecutive_errors} consecutive LLM failures"
                    )

    # Snap predicted head/tail to the document's S2 type vocabulary (S3-style).
    if snap:
        model = None
        if retriever is not None:
            if getattr(retriever, "_model", None) is None:
                from sentence_transformers import SentenceTransformer
                retriever._model = SentenceTransformer(retriever.model_name)
            model = retriever._model
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
        by_doc: dict[str, list[RelationResult]] = defaultdict(list)
        for r in relations:
            by_doc[r.source_doc_id].append(r)
        snapped: list[RelationResult] = []
        for did, rs in by_doc.items():
            tris = [(r.head, r.relation, r.tail) for r in rs]
            for h, rel, t in snap_doc_triples(tris, sorted(set(doc_types.get(did, []))), model):
                snapped.append(RelationResult(head=h, relation=rel, tail=t, source_doc_id=did))
        relations = snapped

    seen: set[tuple[str, str, str, str]] = set()
    unique: list[RelationResult] = []
    for r in relations:
        key = (r.source_doc_id, r.head, r.relation, r.tail)
        if key not in seen:
            seen.add(key)
            unique.append(r)

    logger.info(
        "S5 done: %d relations from %d documents, %d errors",
        len(unique), len(documents), errors,
    )
    return unique


def _infer_domain(
    types: list[str],
    sample_passages: list[str],
    client: LLMClient,
) -> str | None:
    """One LLM call to infer domain_description."""
    messages = build_s5_domain_messages(types, sample_passages)
    response = client.chat_json(messages)
    desc = response.get("domain_description", "").strip()
    return desc if desc else None


def _call_document(
    doc: dict[str, str],
    typed_terms: list[tuple[str, str]],
    domain_description: str | None,
    client: LLMClient,
    examples: list[dict[str, Any]] | None,
    doc_types: list[str] | None = None,
    prompt_v2: bool = False,
) -> list[RelationResult]:
    messages = build_s5_messages(
        passage=doc["text"],
        typed_terms=typed_terms,
        domain_description=domain_description,
        examples=examples,
        doc_types=doc_types,
        prompt_v2=prompt_v2,
    )
    response = client.chat_json(messages)

    found: list[RelationResult] = []
    for entry in response.get("triples", []):
        subject = (entry.get("subject") or "").strip()
        relation = (entry.get("relation") or "").strip()
        obj = (entry.get("object") or "").strip()
        if subject and relation and obj:
            found.append(RelationResult(
                head=subject,
                relation=relation,
                tail=obj,
                source_doc_id=str(doc["id"]),
            ))
    return found
