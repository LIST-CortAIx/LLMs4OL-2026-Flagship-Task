"""Create a reproducible 80/20 train/test split for proper evaluation.

Reads ``data/train_task_a.json`` and produces:

  data/splits/train.json              — 80% of documents (training)
  data/splits/test.json               — 20% of documents (evaluation)
  data/splits/train_corpus.jsonl      — corpus for S0 index (train docs only)
  data/splits/test_corpus.jsonl       — corpus for pipeline inference
  data/splits/test_gold/              — per-step gold files for test set
    submission_gold.json
    s1_gold.json  s2_gold.json  s3_gold.json  s4_gold.json  s5_gold.json

The split is stratified to preserve the ratio of documents that have
instance-of triples (S1/S3 gold) vs. only is-a/non-taxonomic triples.

Usage
-----
    python split_data.py
    python split_data.py --seed 42 --test-ratio 0.2
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import correct_term

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TAXONOMIC_RELATIONS = {"is-a"}
INSTANCE_RELATIONS  = {"instance-of"}


def _write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_gold(entries: list[dict], gold_dir: Path, correct_terms: bool = False) -> None:
    """Build per-step gold files from a list of training entries."""
    all_terms: dict[str, set[str]] = defaultdict(set)
    all_types: dict[str, set[str]] = defaultdict(set)
    term_typings: dict[str, set[str]] = defaultdict(set)
    tax_docs: dict[tuple[str, str], set[str]] = defaultdict(set)
    nontax_docs: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    n_corrected = 0

    for entry in entries:
        doc_id = entry["id"]
        text = entry["context"]
        for triple in entry["primitive-ontology-triples"]:
            if len(triple) != 3:
                continue
            head, relation, tail = triple
            if relation in TAXONOMIC_RELATIONS:
                all_types[head].add(doc_id)
                all_types[tail].add(doc_id)
                tax_docs[(tail, head)].add(doc_id)
            elif relation in INSTANCE_RELATIONS:
                if correct_terms:
                    corrected = correct_term(head, text)
                    if corrected != head:
                        n_corrected += 1
                    head = corrected
                all_types[tail].add(doc_id)
                all_terms[head].add(doc_id)
                term_typings[head].add(tail)
            else:
                nontax_docs[(head, relation, tail)].add(doc_id)

    if correct_terms:
        logger.info("  Gold correction: %d terms replaced with text form", n_corrected)

    gold_dir.mkdir(parents=True, exist_ok=True)

    # submission_gold
    submission_gold = [
        {"id": e["id"], "primitive-ontology-triples": e["primitive-ontology-triples"]}
        for e in entries
    ]
    _write(gold_dir / "submission_gold.json", submission_gold)

    # S1
    s1 = {"terms": [
        {"text": t, "source_doc_ids": sorted(d)}
        for t, d in sorted(all_terms.items())
    ]}
    _write(gold_dir / "s1_gold.json", s1)

    # S2
    s2 = {"types": [
        {"text": t, "source_doc_ids": sorted(d)}
        for t, d in sorted(all_types.items())
    ]}
    _write(gold_dir / "s2_gold.json", s2)

    # S3
    s3 = {"term_typings": [
        {"term": t, "types": sorted(types), "source_doc_ids": sorted(all_terms[t])}
        for t, types in sorted(term_typings.items())
    ]}
    _write(gold_dir / "s3_gold.json", s3)

    # S4
    s4 = {"taxonomic_relations": [
        {"parent": p, "child": c, "source_doc_ids": sorted(d)}
        for (p, c), d in sorted(tax_docs.items())
    ]}
    _write(gold_dir / "s4_gold.json", s4)

    # S5
    non_tax = [
        {"head": h, "relation": r, "tail": t, "source_doc_ids": sorted(d)}
        for (h, r, t), d in sorted(nontax_docs.items())
    ]
    _write(gold_dir / "s5_gold.json", {"non_taxonomic_relations": non_tax})

    n_io = sum(1 for e in entries
               if any(len(t)==3 and t[1]=="instance-of"
                      for t in e.get("primitive-ontology-triples",[])))
    logger.info("  submission_gold: %d docs", len(entries))
    logger.info("  S1 gold: %d terms from %d docs (%.1f%%)",
                len(s1["terms"]), n_io, 100*n_io/len(entries) if entries else 0)
    logger.info("  S2 gold: %d types", len(s2["types"]))
    logger.info("  S3 gold: %d term-type pairs", sum(len(v) for v in term_typings.values()))
    logger.info("  S4 gold: %d is-a relations", len(s4["taxonomic_relations"]))
    logger.info("  S5 gold: %d non-taxonomic relations", len(non_tax))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/train_task_a.json")
    parser.add_argument("--out-dir", default="data/splits")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--correct-terms", action="store_true",
                        help="Replace gold terms not found in doc text with closest text span")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    logger.info("Loaded %d documents from %s", len(data), args.input)

    # Stratified split: preserve ratio of docs with/without instance-of triples
    has_io_set = {e["id"] for e in data if any(
        len(t)==3 and t[1]=="instance-of" for t in e.get("primitive-ontology-triples",[]))}
    group_io = [e for e in data if e["id"] in has_io_set]
    group_no = [e for e in data if e["id"] not in has_io_set]

    rng = random.Random(args.seed)
    rng.shuffle(group_io)
    rng.shuffle(group_no)

    n_test_io = max(1, int(len(group_io) * args.test_ratio))
    n_test_no = max(1, int(len(group_no) * args.test_ratio))

    test  = group_io[:n_test_io] + group_no[:n_test_no]
    train = group_io[n_test_io:] + group_no[n_test_no:]
    rng.shuffle(test)
    rng.shuffle(train)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Split: %d train (%.0f%%), %d test (%.0f%%)",
                len(train), 100*len(train)/len(data),
                len(test),  100*len(test)/len(data))
    logger.info("  Train: %d with instance-of, %d without",
                len(group_io)-n_test_io, len(group_no)-n_test_no)
    logger.info("  Test:  %d with instance-of, %d without", n_test_io, n_test_no)

    # Write raw splits
    _write(out / "train.json", train)
    _write(out / "test.json", test)
    logger.info("Written: %s/train.json, test.json", out)

    # Train corpus (for S0 index)
    with (out / "train_corpus.jsonl").open("w", encoding="utf-8") as f:
        for e in train:
            f.write(json.dumps({"id": e["id"], "text": e["context"]}, ensure_ascii=False) + "\n")
    logger.info("Written: %s/train_corpus.jsonl", out)

    # Test corpus (pipeline input)
    with (out / "test_corpus.jsonl").open("w", encoding="utf-8") as f:
        for e in test:
            f.write(json.dumps({"id": e["id"], "text": e["context"]}, ensure_ascii=False) + "\n")
    logger.info("Written: %s/test_corpus.jsonl", out)

    # Gold files for the TRAIN set — source of RAG few-shot examples (S3 pair
    # index, etc.). Must be built from train only to avoid test leakage.
    logger.info("Building train gold files (original)...")
    build_gold(train, out / "train_gold", correct_terms=False)
    logger.info("Written: %s/train_gold/", out)

    # Gold files for test set (original, no correction)
    logger.info("Building test gold files (original)...")
    build_gold(test, out / "test_gold", correct_terms=False)
    logger.info("Written: %s/test_gold/", out)

    # Gold files for test set (corrected terms)
    if args.correct_terms:
        logger.info("Building test gold files (corrected)...")
        build_gold(test, out / "test_gold_corrected", correct_terms=True)
        logger.info("Written: %s/test_gold_corrected/", out)


if __name__ == "__main__":
    main()
