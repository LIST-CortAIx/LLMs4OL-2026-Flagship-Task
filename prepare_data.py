"""Prepare data for the LLMs4OL pipeline: build gold files AND retriever indexes.

Builds, for the chosen --scope, the per-step gold files and the S0 retriever indexes
(doc text/terms/types embeddings + S3 pair index):

  whole  (final submission — examples from all training docs)
    data/corpus.jsonl
    data/gold/
    indexes/s0_retriever_full
  split  (development eval — needs data/splits/{train,test}.json from split_data.py)
    data/splits/train_gold/
    data/splits/test_gold/
    indexes/split_s0_retriever

Indexes are always built from the TRAIN portion only (whole = all docs; split =
train.json) so they never leak the test answers.

Usage
-----
    python prepare_data.py --scope whole       # final submission artifacts
    python prepare_data.py --scope split       # dev artifacts (after split_data.py)
    python prepare_data.py --scope both
    python prepare_data.py --scope whole --no-index        # gold only
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TAXONOMIC_RELATIONS = {"is-a"}
INSTANCE_RELATIONS = {"instance-of"}  # S3 only — not S5


def _tok(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def correct_term(term: str, text: str) -> str:
    """Replace *term* with its surface form in *text* (literal or underscore→space)."""
    m = re.search(re.escape(term), text, re.IGNORECASE)
    if m:
        return m.group(0)
    if "_" in term:
        m = re.search(re.escape(term.replace("_", " ")), text, re.IGNORECASE)
        if m:
            return m.group(0)
    return term


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Gold building ────────────────────────────────────────────────────────────

def build_gold_files(entries: list[dict], gold_dir: Path, correct: bool = False) -> None:
    """Derive per-step gold files from primitive-ontology-triples."""
    gold_dir.mkdir(parents=True, exist_ok=True)
    all_terms: dict[str, set[str]] = defaultdict(set)
    all_types: dict[str, set[str]] = defaultdict(set)
    term_typings: dict[str, set[str]] = defaultdict(set)
    tax_docs: dict[tuple[str, str], set[str]] = defaultdict(set)
    nontax_docs: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    n_corrected = 0
    for entry in entries:
        doc_id, text = entry["id"], entry["context"]
        for triple in entry["primitive-ontology-triples"]:
            if len(triple) != 3:
                continue
            head, relation, tail = triple
            if relation in TAXONOMIC_RELATIONS:
                all_types[head].add(doc_id)
                all_types[tail].add(doc_id)
                tax_docs[(tail, head)].add(doc_id)
            elif relation in INSTANCE_RELATIONS:
                if correct:
                    c = correct_term(head, text)
                    if c != head:
                        n_corrected += 1
                    head = c
                all_types[tail].add(doc_id)
                all_terms[head].add(doc_id)
                term_typings[head].add(tail)
            else:
                nontax_docs[(head, relation, tail)].add(doc_id)

    _write(gold_dir / "submission_gold.json",
           [{"id": e["id"], "primitive-ontology-triples": e["primitive-ontology-triples"]} for e in entries])
    _write(gold_dir / "s1_gold.json",
           {"terms": [{"text": t, "source_doc_ids": sorted(d)} for t, d in sorted(all_terms.items())]})
    _write(gold_dir / "s2_gold.json",
           {"types": [{"text": t, "source_doc_ids": sorted(d)} for t, d in sorted(all_types.items())]})
    _write(gold_dir / "s3_gold.json",
           {"term_typings": [{"term": t, "types": sorted(ty), "source_doc_ids": sorted(all_terms[t])}
                             for t, ty in sorted(term_typings.items())]})
    _write(gold_dir / "s4_gold.json",
           {"taxonomic_relations": [{"parent": p, "child": c, "source_doc_ids": sorted(d)}
                                    for (p, c), d in sorted(tax_docs.items())]})
    nontax = [{"head": h, "relation": r, "tail": t, "source_doc_ids": sorted(d)}
              for (h, r, t), d in sorted(nontax_docs.items())]
    _write(gold_dir / "s5_gold.json", {"non_taxonomic_relations": nontax})
    logger.info("  %s%s: %d terms, %d types, %d typings, %d is-a, %d non-tax%s",
                gold_dir.name, " (corrected)" if correct else "",
                len(all_terms), len(all_types), len(term_typings), len(tax_docs), len(nontax),
                f" [{n_corrected} terms corrected]" if correct else "")


# ── Index building ───────────────────────────────────────────────────────────

def build_index(training_path: Path, gold_dir: Path, index_dir: Path,
                model_name: str, correct: bool) -> None:
    """Build the S0 retriever index in one directory:
    doc (text/terms/types) index + S3 pair index, plus the S4 taxonomy-parent
    index for the base (non-corrected) variant only — the corrected index is the
    terms-only variant, and taxonomy parents (is-a types) are read from the base."""
    import s0_retriever
    logger.info("Building index → %s (corrected=%s)", index_dir, correct)
    s0_retriever.build_doc_index(training_path=str(training_path), index_dir=str(index_dir),
                                 model_name=model_name, correct_terms=correct)
    s0_retriever.build(gold_path=str(gold_dir / "s3_gold.json"), index_dir=str(index_dir),
                       model_name=model_name)
    if not correct:
        s0_retriever.build_taxonomy_parent_index(training_path=str(training_path),
                                                 index_dir=str(index_dir), model_name=model_name)


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def do_whole(args) -> None:
    src = Path(args.input)
    entries = _load(src)
    logger.info("WHOLE: %d docs from %s", len(entries), src)
    # corpus.jsonl (S1/S2 input convenience)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    corpus = Path(args.data_dir) / "corpus.jsonl"
    with corpus.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps({"id": e["id"], "text": e["context"]}, ensure_ascii=False) + "\n")
    logger.info("  corpus → %s", corpus)
    build_gold_files(entries, Path("data/gold"), correct=False)
    if not args.no_index:
        build_index(src, Path("data/gold"), Path("indexes/s0_retriever_full"),
                    args.embedding_model, correct=False)


def do_split(args) -> None:
    sd = Path(args.splits_dir)
    train_p, test_p = sd / "train.json", sd / "test.json"
    if not (train_p.exists() and test_p.exists()):
        sys.exit(f"Split files not found in {sd} — run split_data.py first.")
    train, test = _load(train_p), _load(test_p)
    logger.info("SPLIT: %d train, %d test", len(train), len(test))
    build_gold_files(train, sd / "train_gold", correct=False)
    build_gold_files(test, sd / "test_gold", correct=False)
    if not args.no_index:
        build_index(train_p, sd / "train_gold", Path("indexes/split_s0_retriever"),
                    args.embedding_model, correct=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scope", choices=["whole", "split", "both"], default="whole",
                        help="Which artifacts to build (default: whole)")
    parser.add_argument("--input", default="data/train_task_a.json",
                        help="Raw challenge JSON for the whole scope (default: data/train_task_a.json)")
    parser.add_argument("--splits-dir", default="data/splits",
                        help="Directory with train.json/test.json for the split scope")
    parser.add_argument("--data-dir", default="data", help="Output data dir (default: data)")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model for indexes (default: all-MiniLM-L6-v2)")
    parser.add_argument("--no-index", action="store_true", help="Build gold files only (skip indexes)")
    args = parser.parse_args()

    if args.scope in ("whole", "both"):
        do_whole(args)
    if args.scope in ("split", "both"):
        do_split(args)
    logger.info("Done (scope=%s, index=%s).", args.scope, not args.no_index)


if __name__ == "__main__":
    main()
