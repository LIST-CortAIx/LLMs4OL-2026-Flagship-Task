"""Evaluation utilities for each pipeline step.

All per-step evaluations are per-document: for each document, predicted items
are compared against gold items for that document, producing P/R/F1. Results
are aggregated as both macro-average (each document weighted equally) and
micro-average (pooled TP/FP/FN across all documents).

Gold file formats (each entry carries source_doc_ids):
    s1_gold.json  → {"terms": [{"text", "source_doc_ids"}]}
    s2_gold.json  → {"types": [...]}   (no per-doc structure — global only)
    s3_gold.json  → {"term_typings": [{"term", "types", "source_doc_ids"}]}
    s4_gold.json  → {"taxonomic_relations": [{"parent", "child", "source_doc_ids"}]}
    s5_gold.json  → {"non_taxonomic_relations": [{"head", "relation", "tail", "source_doc_ids"}]}
    submission_gold.json → [{"id", "primitive-ontology-triples"}]

All string comparisons are case-insensitive.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _metrics(tp: int, fp: int, fn: int) -> dict[str, Any]:
    if tp == 0 and fp == 0 and fn == 0:  # both pred and gold empty — perfect
        precision, recall, f1 = 1.0, 1.0, 1.0
    else:
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gold": tp + fn,
        "n_pred": tp + fp,
    }


def _doc_metrics(pred: set, gold: set) -> dict[str, Any]:
    tp = len(pred & gold)
    return _metrics(tp, len(pred - gold), len(gold - pred))


# ---------------------------------------------------------------------------
# Grouping helpers — build {doc_id: set_of_items} from each format
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Normalize for comparison: lowercase, underscores → spaces.

    Gold uses underscore-encoded identifiers (frustum_of_cone);
    predictions extract the form from text (frustum of cone).
    Both are normalized to space-separated lowercase for fair comparison.
    """
    return s.lower().replace("_", " ")


def _s1_by_doc(data: dict[str, Any]) -> dict[str, set[str]]:
    by_doc: dict[str, set[str]] = defaultdict(set)
    for t in data.get("terms", []):
        item = _norm(t["text"])
        for did in t.get("source_doc_ids", []):
            by_doc[did].add(item)
    return by_doc


def _s2_by_doc(data: dict[str, Any]) -> dict[str, set[str]]:
    by_doc: dict[str, set[str]] = defaultdict(set)
    for t in data.get("types", []):
        if isinstance(t, str):
            continue  # old flat format has no doc info — skip
        text = _norm(t["text"])
        for did in t.get("source_doc_ids", []):
            by_doc[did].add(text)
    return by_doc


def _s3_by_doc(data: dict[str, Any]) -> dict[str, set[tuple[str, str]]]:
    by_doc: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for tt in data.get("term_typings", []):
        term = _norm(tt["term"])
        for typ in tt.get("types", []):
            pair = (term, _norm(typ))
            for did in tt.get("source_doc_ids", []):
                by_doc[did].add(pair)
    return by_doc


def _s4_by_doc(data: dict[str, Any]) -> dict[str, set[tuple[str, str]]]:
    by_doc: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for r in data.get("taxonomic_relations", []):
        pair = (_norm(r["parent"]), _norm(r["child"]))
        for did in r.get("source_doc_ids", []):
            by_doc[did].add(pair)
    return by_doc


def _s5_by_doc(data: dict[str, Any], doc_id_key: str = "source_doc_ids") -> dict[str, set[tuple[str, str, str]]]:
    """Group S5 relations by document.

    Gold uses ``source_doc_ids`` (list); predictions use ``source_doc_id`` (str).
    Pass ``doc_id_key="source_doc_id"`` for prediction files.
    """
    by_doc: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for r in data.get("non_taxonomic_relations", []):
        triple = (_norm(r["head"]), _norm(r["relation"]), _norm(r["tail"]))
        if doc_id_key == "source_doc_ids":
            for did in r.get("source_doc_ids", []):
                by_doc[did].add(triple)
        else:
            did = r.get("source_doc_id", "")
            if did:
                by_doc[did].add(triple)
    return by_doc


# ---------------------------------------------------------------------------
# Per-document evaluation — one entry per doc, {doc_id: metrics_dict}
# ---------------------------------------------------------------------------

def eval_s1_per_doc(
    predicted: dict[str, Any], gold: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Per-document term extraction F1."""
    gold_by_doc = _s1_by_doc(gold)
    pred_by_doc = _s1_by_doc(predicted)
    all_docs = gold_by_doc.keys() | pred_by_doc.keys()
    return {did: _doc_metrics(pred_by_doc[did], gold_by_doc[did]) for did in all_docs}


def eval_s2_per_doc(
    predicted: dict[str, Any], gold: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Per-document type vocabulary F1."""
    gold_by_doc = _s2_by_doc(gold)
    pred_by_doc = _s2_by_doc(predicted)
    all_docs = gold_by_doc.keys() | pred_by_doc.keys()
    return {did: _doc_metrics(pred_by_doc[did], gold_by_doc[did]) for did in all_docs}


def eval_s3_per_doc(
    predicted: dict[str, Any], gold: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Per-document (term, type) pair F1."""
    gold_by_doc = _s3_by_doc(gold)
    pred_by_doc = _s3_by_doc(predicted)
    all_docs = gold_by_doc.keys() | pred_by_doc.keys()
    return {did: _doc_metrics(pred_by_doc[did], gold_by_doc[did]) for did in all_docs}


def eval_s4_per_doc(
    predicted: dict[str, Any], gold: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Per-document (parent, child) is-a pair F1."""
    gold_by_doc = _s4_by_doc(gold)
    pred_by_doc = _s4_by_doc(predicted)
    all_docs = gold_by_doc.keys() | pred_by_doc.keys()
    return {did: _doc_metrics(pred_by_doc[did], gold_by_doc[did]) for did in all_docs}


def eval_s5_per_doc(
    predicted: dict[str, Any], gold: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Per-document (head, relation, tail) triple F1."""
    gold_by_doc = _s5_by_doc(gold, doc_id_key="source_doc_ids")
    pred_by_doc = _s5_by_doc(predicted, doc_id_key="source_doc_id")
    all_docs = gold_by_doc.keys() | pred_by_doc.keys()
    return {did: _doc_metrics(pred_by_doc[did], gold_by_doc[did]) for did in all_docs}


def eval_submission_per_doc(
    predicted: list[dict[str, Any]], gold: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Per-document primitive-ontology-triple F1 (final submission format)."""
    gold_by_doc = {
        d["id"]: {(_norm(s), _norm(r), _norm(o)) for s, r, o in d.get("primitive-ontology-triples", [])}
        for d in gold
    }
    pred_by_doc = {
        d["id"]: {(_norm(s), _norm(r), _norm(o)) for s, r, o in d.get("primitive-ontology-triples", [])}
        for d in predicted
    }
    all_docs = gold_by_doc.keys() | pred_by_doc.keys()
    return {
        did: _doc_metrics(pred_by_doc.get(did, set()), gold_by_doc.get(did, set()))
        for did in all_docs
    }




# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def macro_average(per_doc: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Unweighted mean of per-document P/R/F1 (each document counts equally)."""
    if not per_doc:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_docs": 0}
    vals = list(per_doc.values())
    n = len(vals)
    return {
        "precision": round(sum(m["precision"] for m in vals) / n, 4),
        "recall":    round(sum(m["recall"]    for m in vals) / n, 4),
        "f1":        round(sum(m["f1"]        for m in vals) / n, 4),
        "n_docs": n,
    }


def micro_average(per_doc: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pooled TP/FP/FN across all documents (larger docs weighted more).

    Empty-empty docs (pred=[] and gold=[]) contribute 1 virtual TP so that
    a correct empty prediction is counted as F1=1.0 in the micro average.
    """
    empty_empty = sum(1 for m in per_doc.values()
                      if m["tp"] == 0 and m["fp"] == 0 and m["fn"] == 0
                      and m["n_gold"] == 0 and m["n_pred"] == 0)
    tp = sum(m["tp"] for m in per_doc.values()) + empty_empty
    fp = sum(m["fp"] for m in per_doc.values())
    fn = sum(m["fn"] for m in per_doc.values())
    result = _metrics(tp, fp, fn)
    result["n_docs"] = len(per_doc)
    return result


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_summary(
    per_doc: dict[str, dict[str, Any]],
    label: str = "",
) -> None:
    """Print only macro and micro averages — used in pipeline logs."""
    macro = macro_average(per_doc)
    micro = micro_average(per_doc)
    if label:
        print(f"\n{label}")
    print(
        f"  Macro-avg ({macro['n_docs']} docs): "
        f"P={macro['precision']:.4f}  R={macro['recall']:.4f}  F1={macro['f1']:.4f}"
    )
    print(
        f"  Micro-avg ({micro['n_docs']} docs): "
        f"P={micro['precision']:.4f}  R={micro['recall']:.4f}  F1={micro['f1']:.4f}"
        f"  TP={micro['tp']}  FP={micro['fp']}  FN={micro['fn']}"
    )


def print_eval(
    per_doc: dict[str, dict[str, Any]],
    label: str = "",
    top_n: int = 20,
    output_path: str | None = None,
) -> None:
    """Print per-document table (worst first) followed by macro and micro averages.

    Args:
        per_doc: {doc_id: metrics_dict} from any eval_*_per_doc function.
        label: Section header printed before the table.
        top_n: Number of worst documents to show in the table.
        output_path: If given, save full per-doc metrics dict as JSON here.
    """
    if output_path:
        with open(output_path, "w") as fh:
            json.dump(per_doc, fh, indent=2)
        print(f"  Per-doc metrics saved → {output_path}")

    rows = sorted(per_doc.items(), key=lambda kv: kv[1]["f1"])
    n_show = min(top_n, len(rows))

    if label:
        print(f"\n{label}")
    print(f"  Per-document results — worst first ({n_show}/{len(rows)} shown)")
    print(f"  {'doc_id':<36}  {'P':>6}  {'R':>6}  {'F1':>6}  {'|Gold|':>7}  {'|Pred|':>7}")
    print(f"  {'-'*36}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}")
    for did, m in rows[:top_n]:
        print(
            f"  {did:<36}  {m['precision']:>6.3f}  {m['recall']:>6.3f}"
            f"  {m['f1']:>6.3f}  {m['n_gold']:>7}  {m['n_pred']:>7}"
        )
    if len(rows) > top_n:
        rest = rows[top_n:]
        avg_f1 = sum(m["f1"] for _, m in rest) / len(rest)
        print(f"  … {len(rest)} more docs (avg F1 of remaining: {avg_f1:.3f})")

    print_summary(per_doc)


# ---------------------------------------------------------------------------
# Graph Similarity Metric (organizer metric — LLMs4OL 2026)
# ---------------------------------------------------------------------------

def graph_similarity(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    modes: tuple[str, ...] = ("exact", "fuzzy", "semantic"),
) -> dict[str, Any]:
    """Official organizer Graph Similarity, scored PER DOCUMENT (matched by id) and averaged.

    Delegates to the vendored organizer code in ``ontology_learning.official_eval`` so the
    numbers match the leaderboard exactly. Each of exact/fuzzy/semantic returns
    {edge_f1, neighborhood_similarity, taxonomy_similarity, graph_similarity}.

    Per-doc (not global pooling): the fuzzy/semantic modes run a Hungarian alignment on a
    ``len(pred)×len(gold)`` matrix — pooled globally (~20k×20k) that's infeasible, and the
    submission/gold are keyed by doc id, so the metric is averaged over per-doc scores.

    Args:
        predicted / gold: lists of {"id", "primitive-ontology-triples": [[s,r,o], ...]}.
        modes: which match modes to compute. Drop "semantic" to avoid the nomic-embed model.
    """
    from ontology_learning.official_eval import graph_similarity as _og

    fns = {"exact": _og.exact_match, "fuzzy": _og.fuzzy_match, "semantic": _og.semantic_match}

    def _by_id(docs):
        out: dict[str, list[tuple]] = {}
        for d in docs:
            out[d["id"]] = [tuple(t) for t in d.get("primitive-ontology-triples", []) if len(t) == 3]
        return out

    pred_by, gold_by = _by_id(predicted), _by_id(gold)
    ids = sorted(set(pred_by) | set(gold_by))

    from collections import defaultdict
    sums = {m: defaultdict(float) for m in modes}
    for did in ids:
        g, p = gold_by.get(did, []), pred_by.get(did, [])
        for m in modes:
            for k, v in fns[m](g, p).items():
                sums[m][k] += v

    n = len(ids) or 1
    result: dict[str, Any] = {
        "n_docs": len(ids),
        "n_pred_triples": sum(len(v) for v in pred_by.values()),
        "n_gold_triples": sum(len(v) for v in gold_by.values()),
    }
    for m in modes:
        result[f"{m}_match"] = {k: round(v / n, 4) for k, v in sums[m].items()}
    return result


def print_graph_similarity(scores: dict[str, Any], label: str = "") -> None:
    """Print the official per-doc-averaged graph similarity (exact/fuzzy/semantic)."""
    if label:
        print(f"\n{label}")
    print(f"  docs={scores['n_docs']}  triples pred={scores['n_pred_triples']} gold={scores['n_gold_triples']}")
    print(f"  {'mode':9s} {'edge_f1':>8s} {'neighbor':>9s} {'taxonomy':>9s} {'graph_sim':>10s}")
    for m in ("exact", "fuzzy", "semantic"):
        key = f"{m}_match"
        if key not in scores:
            continue
        s = scores[key]
        print(f"  {m:9s} {s['edge_f1']:>8.4f} {s['neighborhood_similarity']:>9.4f} "
              f"{s['taxonomy_similarity']:>9.4f} {s['graph_similarity']:>10.4f}")


def print_metrics(metrics: dict[str, Any], label: str = "") -> None:
    """Print a single global metrics dict (used for S2)."""
    if label:
        print(f"\n{label}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}"
          f"  |Gold|={metrics['n_gold']}  |Pred|={metrics['n_pred']}")
