"""Build SFT (chat) data for fine-tuning S5 non-taxonomic relation extraction.

Each example: system = the S5 extraction prompt (prompt_v2), user = the doc's S2 type
vocabulary + S3 typed terms + passage (+ optional retrieved few-shot demos), assistant =
the gold relations JSON. Relation-free docs are included with `{"triples": []}` (teaches
abstention from the true base rate).

Two modes:
  * default (no --few-shot): zero-context; with few relation-bearing docs the model tends
    to collapse to always-empty.
  * --few-shot (RAG-style, recommended): inject the k nearest training docs as examples
    (full-pool, like the zero-shot inference) so the model learns to use the examples
    rather than memorize. Examples for a train doc exclude itself.

Input formatting matches build_s5_messages(prompt_v2=True[, examples]); serving/eval of
the fine-tuned model must use the same flags (few-shot, full-pool, k-examples).

Usage:
    # split-train data (development):
    python scripts/prepare_s5_finetune.py --scope split --few-shot --oversample-pos 5 \
        --out-dir data/s5_ft_split
    # whole-train data (final submission):
    python scripts/prepare_s5_finetune.py --scope whole --few-shot --out-dir data/s5_ft_whole
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from ontology_learning.prompts import build_s5_messages


def _by_doc_types(s2_path):
    m = defaultdict(list)
    for t in json.loads(Path(s2_path).read_text())["types"]:
        for d in t.get("source_doc_ids", []):
            m[str(d)].append(t["text"])
    return m


def _by_doc_terms(s3_path):
    m = defaultdict(list)
    for tt in json.loads(Path(s3_path).read_text())["term_typings"]:
        for d in tt.get("source_doc_ids", []):
            for ty in tt.get("types", []):
                m[str(d)].append((tt["term"], ty))
    return m


def _by_doc_triples(s5_path):
    m = defaultdict(list)
    for r in json.loads(Path(s5_path).read_text())["non_taxonomic_relations"]:
        doc_ids = r.get("source_doc_ids") or ([r["source_doc_id"]] if r.get("source_doc_id") else [])
        for d in doc_ids:
            m[str(d)].append({"subject": r["head"], "relation": r["relation"], "object": r["tail"]})
    return m


def build(docs_path, gold_dir, prompt_v2, retriever=None, pool=None, k=0, with_text=True):
    docs = json.loads(Path(docs_path).read_text())
    types = _by_doc_types(f"{gold_dir}/s2_gold.json")
    terms = _by_doc_terms(f"{gold_dir}/s3_gold.json")
    triples = _by_doc_triples(f"{gold_dir}/s5_gold.json")
    if retriever is not None and pool and k > 0:
        from ontology_learning.steps.s5_relations import retrieve_examples
    out = []
    for d in docs:
        did = str(d["id"])
        doc_types = sorted(set(types.get(did, [])))
        examples = None
        if retriever is not None and pool and k > 0:
            examples = retrieve_examples(retriever, pool, d.get("context", ""), k,
                                         exclude_id=did, with_text=with_text)
        msgs = build_s5_messages(
            passage=d.get("context", ""), typed_terms=terms.get(did, []),
            domain_description=None, examples=examples,
            doc_types=doc_types or None, prompt_v2=prompt_v2,
        )
        answer = json.dumps({"triples": triples.get(did, [])}, ensure_ascii=False)
        out.append({"messages": msgs + [{"role": "assistant", "content": answer}]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", choices=["split", "whole"], default="split")
    ap.add_argument("--out-dir", default="data/s5_ft")
    ap.add_argument("--no-prompt-v2", action="store_true")
    ap.add_argument("--few-shot", action="store_true",
                    help="RAG-style: inject k nearest full-pool demos into each prompt")
    ap.add_argument("--index-dir", default=None, help="Doc retriever index (default per scope)")
    ap.add_argument("--k-examples", type=int, default=10)
    ap.add_argument("--no-examples-text", action="store_true", help="Drop demo passages (types→relations only)")
    ap.add_argument("--oversample-pos", type=int, default=1,
                    help="Duplicate relation-bearing TRAIN examples N times (fight majority-collapse)")
    ap.add_argument("--max-empty", type=int, default=0, help="Cap empty TRAIN examples (0 = keep all)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pv2 = not args.no_prompt_v2

    if args.scope == "split":
        sets = [("train", "data/splits/train.json", "data/splits/train_gold"),
                ("val", "data/splits/test.json", "data/splits/test_gold")]
        train_gold, index_default = "data/splits/train_gold", "indexes/split_s0_retriever"
    else:
        sets = [("train", "data/train_task_a.json", "data/gold")]
        train_gold, index_default = "data/gold", "indexes/s0_retriever_full"

    retriever = pool = None
    if args.few_shot:
        from ontology_learning.steps.s0_retriever import load_doc_index
        from ontology_learning.steps.s5_relations import load_full_example_pool
        retriever = load_doc_index(args.index_dir or index_default)
        pool = load_full_example_pool(f"{train_gold}/s2_gold.json", f"{train_gold}/s5_gold.json")
        print(f"Few-shot RAG: index={args.index_dir or index_default}, pool={len(pool)} train docs, "
              f"k={args.k_examples}, text={not args.no_examples_text}")

    for name, docs_path, gold_dir in sets:
        ex = build(docs_path, gold_dir, pv2, retriever, pool,
                   args.k_examples if args.few_shot else 0, not args.no_examples_text)
        if name == "train" and (args.oversample_pos > 1 or args.max_empty):
            rng = random.Random(args.seed)
            pos = [e for e in ex if json.loads(e["messages"][-1]["content"])["triples"]]
            emp = [e for e in ex if not json.loads(e["messages"][-1]["content"])["triples"]]
            if args.max_empty and len(emp) > args.max_empty:
                emp = rng.sample(emp, args.max_empty)
            ex = pos * args.oversample_pos + emp
            rng.shuffle(ex)
        npos = sum(1 for e in ex if json.loads(e["messages"][-1]["content"])["triples"])
        path = out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for e in ex:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"{name}: {len(ex)} examples ({npos} relation-bearing, {len(ex)-npos} empty) → {path}")


if __name__ == "__main__":
    main()
