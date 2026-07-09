"""Step-by-step pipeline runner.

Runs one step at a time, reads its input from disk, and writes its output to
disk.  S0 is CPU-only (no LLM); S1S2, S3–S5 call the vLLM server.
Outputs can be evaluated against per-step gold files with the ``eval``
subcommand.

File formats
------------
    corpus (--input for s1s2, --corpus for s5):  JSONL, one {id, text} per line
    s1_output.json:  {"terms": [{"text": ..., "source_doc_ids": [...], "context_sentence": ...}]}
    s2_output.json:  {"types": [{"text": ..., "source_doc_ids": [...]}]}
    s3_output.json:  {"term_typings": [{"term": ..., "types": [...], "source_doc_ids": [...]}]}
    s4_output.json:  {"taxonomic_relations": [{"parent": ..., "child": ..., "source_doc_ids": [...]}]}
    s5_output.json:  {"non_taxonomic_relations": [{"head": ..., "relation": ..., "tail": ..., "source_doc_id": ...}]}

Usage
-----
    # S0: build embedding index (S3 RAG) + embedding doc index (S1S2 few-shot)
    python run_step.py s0 \\
        --gold data/gold/s3_gold.json --index-dir outputs/s0_retriever \\
        --training-data data/train_task_a.json

    # S1: per-document term extraction with RAG few-shot
    python run_step.py s1 \\
        --input data/corpus.jsonl --index-dir outputs/s0_retriever \\
        --output outputs/s1_output.json

    # S2: per-document type extraction with RAG few-shot
    python run_step.py s2 \\
        --input data/corpus.jsonl --index-dir outputs/s0_retriever \\
        --output outputs/s2_output.json

    python run_step.py s3 \\
        --s1-input outputs/s1_output.json --index-dir outputs/s0_retriever \\
        --output outputs/s3_output.json
    python run_step.py s4 --input outputs/s3_output.json --output outputs/s4_output.json
    python run_step.py s5 \\
        --corpus data/corpus.jsonl --s1-input outputs/s1_output.json \\
        --input outputs/s3_output.json --output outputs/s5_output.json

    python run_step.py eval --step s3 \\
        --predictions outputs/s3_output.json --gold data/gold/s3_gold.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import LLMClient
import s2_type_extraction

def _selected_scope() -> str:
    """Pipeline scope from --scope (any position), else $SCOPE, else 'whole'."""
    for i, a in enumerate(sys.argv):
        if a == "--scope" and i + 1 < len(sys.argv):
            return sys.argv[i + 1].strip().lower()
        if a.startswith("--scope="):
            return a.split("=", 1)[1].strip().lower()
    return os.environ.get("SCOPE", "whole").strip().lower()


# --scope split loads config.split.yaml; default (whole) loads config.yaml. The data,
# index and classifier paths are then read from the selected config automatically.
_SCOPE = _selected_scope()
_CONFIG_FILE = Path(__file__).resolve().parent / (
    "config.split.yaml" if _SCOPE == "split" else "config.yaml"
)


def _load_config() -> dict[str, str]:
    """Read key: value pairs from config.yaml (no pyyaml dependency needed)."""
    cfg: dict[str, str] = {}
    if _CONFIG_FILE.exists():
        for line in _CONFIG_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and ":" in line:
                key, _, val = line.partition(":")
                cfg[key.strip()] = val.split("#")[0].strip()
    return cfg


_CFG = _load_config()


def _cfg_bool(key: str, default: bool) -> bool:
    value = _CFG.get(key, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"config.yaml: {key} must be a boolean, got {value!r}")


def _cfg_optional_int(key: str) -> int | None:
    value = _CFG.get(key, "").strip()
    return int(value) if value else None


from models import (
    ExtractedTerm,
    ExtractedType,
    PipelineOutput,
    RelationResult,
    TaxonomyResult,
    TermTypingResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

TOY_DOCUMENTS = [
    {
        "id": "doc1",
        "text": (
            "In a smart home system, sensors monitor environmental conditions. "
            "A temperature sensor measures room temperature. "
            "A motion sensor detects movement and triggers the alarm system. "
            "The smart thermostat receives temperature readings from the temperature sensor "
            "and adjusts the heating system. "
            "A mobile app allows the user to control the smart thermostat remotely."
        ),
    }
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _common_args(parser: argparse.ArgumentParser, require_output: bool = True) -> None:
    parser.add_argument("--scope", choices=["whole", "split"], default=_SCOPE,
                        help="Config scope: whole (config.yaml) or split (config.split.yaml); "
                             "picks data/index/classifier paths automatically")
    parser.add_argument("--model",
                        default=os.environ.get("LLM_MODEL") or _CFG.get("model"),
                        help="Model path as registered in the vLLM server (default: $LLM_MODEL or config.yaml)")
    parser.add_argument("--base-url",
                        default=os.environ.get("LLM_BASE_URL") or _CFG.get("base_url"),
                        help="vLLM server URL (default: $LLM_BASE_URL or config.yaml)")
    parser.add_argument("--api-key",
                        default=os.environ.get("LLM_API_KEY") or _CFG.get("api_key", "EMPTY"),
                        help="Endpoint API key (default: $LLM_API_KEY or config.yaml)")
    parser.add_argument("--profile", default=_CFG.get("llm_profile", "qwen"),
                        choices=["qwen", "gpt-oss", "vllm", "openai"],
                        help="Backend profile (default: qwen = our vLLM model)")
    parser.add_argument("--insecure", action="store_true",
                        default=str(_CFG.get("verify_ssl", "true")).lower() == "false",
                        help="Skip TLS verification (self-signed https, e.g. Minimax/LiteLLM)")
    parser.add_argument("--reasoning-effort",
                        default=(_CFG.get("reasoning_effort") or None),
                        choices=["low", "medium", "high"],
                        help="gpt-oss profile reasoning level")
    if require_output:
        parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--max-tokens", type=int,
                        default=int(_CFG.get("max_tokens", 32768)))
    parser.add_argument("--temperature", type=float,
                        default=float(_CFG.get("temperature", 1.0)))
    parser.add_argument("--top-p", type=float, default=float(_CFG.get("top_p", 0.95)))
    parser.add_argument("--top-k", type=int, default=int(_CFG.get("top_k", 20)))
    parser.add_argument("--min-p", type=float, default=float(_CFG.get("min_p", 0.0)))
    parser.add_argument("--presence-penalty", type=float,
                        default=float(_CFG.get("presence_penalty", 1.5)))
    parser.add_argument("--repetition-penalty", type=float,
                        default=float(_CFG.get("repetition_penalty", 1.0)))
    parser.add_argument("--workers", type=int,
                        default=int(_CFG.get("workers", 4)))
    _thinking_default = _CFG.get("enable_thinking", "true").lower() == "false"
    parser.add_argument("--no-thinking", action="store_true", default=_thinking_default,
                        help="Disable Qwen3 thinking tokens (chat_template_kwargs).")
    parser.add_argument("--llm-timeout", type=float,
                        default=float(_CFG.get("llm_timeout", 900)),
                        help="HTTP timeout per LLM attempt in seconds (default: 900)")
    _tb = _CFG.get("thinking_budget", "")
    parser.add_argument("--thinking-budget", type=int,
                        default=int(_tb) if _tb else None,
                        help="vLLM thinking token budget: forces </think> after N tokens")


def _make_client(args: argparse.Namespace) -> LLMClient:
    if not args.model:
        sys.exit("--model is required (or set $LLM_MODEL)")
    if not args.base_url:
        sys.exit("--base-url is required (or set $LLM_BASE_URL / config.yaml)")
    return LLMClient(
        base_url=args.base_url,
        model=args.model,
        api_key=getattr(args, "api_key", "EMPTY"),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        json_mode=True,
        enable_thinking=not args.no_thinking,
        thinking_budget=getattr(args, "thinking_budget", None),
        timeout=args.llm_timeout,
        profile=getattr(args, "profile", "qwen"),
        verify_ssl=not getattr(args, "insecure", False),
        reasoning_effort=getattr(args, "reasoning_effort", None),
    )


def _write(path: str, data: object) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Written → %s", out)


def _load_corpus(path: str) -> list[dict]:
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def _load_terms(s1_path: str) -> list[ExtractedTerm]:
    data = json.loads(Path(s1_path).read_text())
    return [
        ExtractedTerm(
            text=t["text"],
            source_doc_ids=t.get("source_doc_ids", []),
            context_sentence=t.get("context_sentence", ""),
        )
        for t in data["terms"]
    ]


def _load_types(s2_path: str) -> list[ExtractedType]:
    data = json.loads(Path(s2_path).read_text())
    return [
        ExtractedType(
            text=t if isinstance(t, str) else t["text"],
            source_doc_ids=([] if isinstance(t, str) else t.get("source_doc_ids", [])),
        )
        for t in data["types"]
    ]


def _load_term_typings(s3_path: str) -> list[TermTypingResult]:
    data = json.loads(Path(s3_path).read_text())
    return [TermTypingResult(term=tt["term"], types=tt["types"],
                             source_doc_ids=tt.get("source_doc_ids", []))
            for tt in data["term_typings"]]


def _load_taxonomy(s4_path: str) -> list[TaxonomyResult]:
    data = json.loads(Path(s4_path).read_text())
    return [TaxonomyResult(parent=r["parent"], child=r["child"],
                           source_doc_ids=r.get("source_doc_ids", []))
            for r in data["taxonomic_relations"]]


def _load_nontaxonomy(s5_path: str) -> list[RelationResult]:
    data = json.loads(Path(s5_path).read_text())
    return [RelationResult(head=r["head"], relation=r["relation"], tail=r["tail"],
                           source_doc_id=r.get("source_doc_id", ""))
            for r in data["non_taxonomic_relations"]]


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def run_s1(args: argparse.Namespace) -> None:
    import s0_retriever, s1_term_extraction

    if getattr(args, "toy", False):
        documents = TOY_DOCUMENTS
        logger.info("Using toy corpus (%d documents)", len(documents))
    else:
        if not args.input:
            sys.exit("--input required for s1 (JSONL corpus) unless --toy is set")
        documents = _load_corpus(args.input)
        logger.info("Loaded %d documents from %s", len(documents), args.input)

    doc_index = None
    index_dir = getattr(args, "index_dir", None)
    retriever_mode = getattr(args, "retriever_mode", "text")
    if index_dir and Path(index_dir).exists():
        if (Path(index_dir) / "doc_embeddings.npy").exists():
            doc_index = s0_retriever.load_doc_index(index_dir, retriever_mode=retriever_mode)

    result = s1_term_extraction.run(
        documents,
        client=_make_client(args),
        workers=args.workers,
        doc_index=doc_index,
        k_examples=args.k_examples,
        k_empty=getattr(args, "k_empty", 0),
        example_selection=getattr(args, "example_selection", "similarity_all_docs"),
        classifier_path=getattr(args, "classifier", None),
        cot=getattr(args, "cot", False),
    )

    data = {
        "terms": [
            {"text": t.text,
             "source_doc_ids": t.source_doc_ids,
             "context_sentence": t.context_sentence}
            for t in result
        ]
    }
    _write(args.output, data)

    print(f"\n=== S1 ===")
    print(f"Terms : {len(result)}")
    for t in result[:15]:
        print(f"  {t.text!r}")
    if len(result) > 15:
        print(f"  ... ({len(result) - 15} more)")


def run_s0(args: argparse.Namespace) -> None:
    from sentence_transformers import SentenceTransformer
    import s0_retriever

    model = SentenceTransformer(args.embedding_model)
    logger.info("S0: building embedding index from %s → %s", args.gold, args.index_dir)
    s0_retriever.build(
        gold_path=args.gold,
        index_dir=args.index_dir,
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        _model=model,
    )

    if getattr(args, "training_data", None):
        logger.info("S0: building doc index from %s → %s", args.training_data, args.index_dir)
        s0_retriever.build_doc_index(
            training_path=args.training_data,
            index_dir=args.index_dir,
            model_name=args.embedding_model,
            batch_size=args.batch_size,
            correct_terms=getattr(args, "correct_terms", False),
            _model=model,
        )
        logger.info("S0: building taxonomy parent index from %s → %s", args.training_data, args.index_dir)
        s0_retriever.build_taxonomy_parent_index(
            training_path=args.training_data,
            index_dir=args.index_dir,
            model_name=args.embedding_model,
            batch_size=args.batch_size,
            _model=model,
        )

    print(f"\n=== S0 ===")
    print(f"Embedding index written to {args.index_dir}")
    if getattr(args, "training_data", None):
        print(f"Doc index + taxonomy parent index written to {args.index_dir}")


def run_s1s2(args: argparse.Namespace) -> None:
    import s0_retriever, s1_term_extraction, s2_type_extraction

    if getattr(args, "toy", False):
        documents = TOY_DOCUMENTS
        logger.info("Using toy corpus (%d documents)", len(documents))
    else:
        if not args.input:
            sys.exit("--input required for s1s2 (JSONL corpus) unless --toy is set")
        documents = _load_corpus(args.input)
        logger.info("Loaded %d documents from %s", len(documents), args.input)

    doc_index = None
    index_dir = getattr(args, "index_dir", None)
    retriever_mode = getattr(args, "retriever_mode", "text")
    if index_dir and Path(index_dir).exists():
        emb_path = Path(index_dir) / "doc_embeddings.npy"
        if emb_path.exists():
            doc_index = s0_retriever.load_doc_index(index_dir, retriever_mode=retriever_mode)
        else:
            logger.warning(
                "S1/S2: no doc embedding index in %s — run s0 --training-data to build it",
                index_dir,
            )

    client = _make_client(args)

    # S1: per-document term extraction with RAG few-shot
    terms = s1_term_extraction.run(
        documents,
        client=client,
        workers=args.workers,
        doc_index=doc_index,
        k_examples=args.k_examples_s1,
        k_empty=getattr(args, "k_empty_s1", 0),
        example_selection=getattr(args, "example_selection", "similarity_all_docs"),
        classifier_path=getattr(args, "classifier", None),
    )

    _write(args.s1_output, {
        "terms": [
            {"text": t.text, "source_doc_ids": t.source_doc_ids, "context_sentence": t.context_sentence}
            for t in terms
        ]
    })

    # S2: per-document type extraction with RAG few-shot (independent of S1)
    types = s2_type_extraction.run(
        documents,
        client=client,
        workers=args.workers,
        doc_index=doc_index,
        k_examples=args.k_examples_s2,
    )

    _write(args.s2_output, {"types": [{"text": t.text, "source_doc_ids": t.source_doc_ids} for t in types]})

    print(f"\n=== S1S2 ===")
    print(f"Terms : {len(terms)}")
    print(f"Types : {len(types)}")
    for t in terms[:10]:
        print(f"  {t.text!r}")
    if len(terms) > 10:
        print(f"  ... ({len(terms) - 10} more)")
    print(f"  types sample: {[t.text for t in types[:10]]}")


def run_s2(args: argparse.Namespace) -> None:
    import s0_retriever, s2_type_extraction

    if getattr(args, "toy", False):
        documents = TOY_DOCUMENTS
    else:
        if not args.input:
            sys.exit("--input required for s2 (JSONL corpus) unless --toy is set")
        documents = _load_corpus(args.input)
        logger.info("Loaded %d documents from %s", len(documents), args.input)

    doc_index = None
    index_dir = getattr(args, "index_dir", None)
    if index_dir and Path(index_dir).exists():
        if (Path(index_dir) / "doc_embeddings.npy").exists():
            doc_index = s0_retriever.load_doc_index(index_dir)

    result = s2_type_extraction.run(
        documents,
        client=_make_client(args),
        workers=args.workers,
        doc_index=doc_index,
        k_examples=args.k_examples,
        example_selection=getattr(args, "s2_example_selection", "similarity"),
        example_seed=getattr(args, "s2_example_seed", None),
        cot=getattr(args, "cot", False),
    )

    data = {"types": [{"text": t.text, "source_doc_ids": t.source_doc_ids} for t in result]}
    _write(args.output, data)

    print(f"\n=== S2 ===")
    print(f"Types : {len(result)}")


def run_s3(args: argparse.Namespace) -> None:
    import s0_retriever, s3_term_typing

    terms = _load_terms(args.s1_input)
    retriever = s0_retriever.load(args.index_dir)

    context_mode = getattr(args, "context", "full")
    documents = None
    if context_mode != "none":
        if not getattr(args, "corpus", None):
            sys.exit("--corpus is required when --context != none (full-doc grounding)")
        documents = _load_corpus(args.corpus)

    type_vocab: set[str] | None = None
    if getattr(args, "s2_input", None) and not getattr(args, "no_snap", False):
        s2_data = json.load(open(args.s2_input))
        type_vocab = {
            (t["text"] if isinstance(t, dict) else t).strip().lower()
            for t in s2_data.get("types", [])
        }
        logger.info("S3: %d terms, k=%d, context=%s, type_vocab=%d types",
                    len(terms), args.k, context_mode, len(type_vocab))
    else:
        logger.info("S3: %d terms, k=%d, context=%s (no type_vocab — free assignment)",
                    len(terms), args.k, context_mode)

    result = s3_term_typing.run(
        terms,
        client=_make_client(args),
        retriever=retriever,
        k=args.k,
        batch_size=args.batch_size,
        workers=args.workers,
        type_vocab=type_vocab,
        documents=documents,
        context_mode=context_mode,
        constrained=getattr(args, "constrained", False),
        lenient=getattr(args, "lenient", False),
        augment_frequent=getattr(args, "augment_frequent", 0),
        hybrid=getattr(args, "hybrid", False),
        hybrid_alpha=getattr(args, "hybrid_alpha", 0.5),
    )

    data = {"term_typings": [{"term": tt.term, "types": tt.types, "source_doc_ids": tt.source_doc_ids} for tt in result]}
    _write(args.output, data)

    assigned = [tt for tt in result if tt.types]
    print(f"\n=== S3 ===")
    print(f"Typed {len(assigned)}/{len(result)} terms")
    for tt in assigned[:10]:
        print(f"  {tt.term!r:35s} → {tt.types}")
    if len(assigned) > 10:
        print(f"  ... ({len(assigned) - 10} more)")


def run_s4(args: argparse.Namespace) -> None:
    import s4_taxonomy

    depth_pass = _cfg_bool("s4_depth_pass", False)
    if getattr(args, "depth_pass", False):
        depth_pass = True
    if getattr(args, "no_depth_pass", False):
        depth_pass = False

    if getattr(args, "embedding", False):
        import s0_retriever
        if not args.input:
            sys.exit("--input is required when --embedding is set (S3 output)")
        term_typings = _load_term_typings(args.input)
        logger.info("S4 embedding: %d term typings", len(term_typings))
        index_dir = getattr(args, "index_dir", None)
        if not index_dir:
            sys.exit("--index-dir is required when --embedding is set (built by s0 --training-data)")
        taxonomy_index = s0_retriever.load_taxonomy_parent_index(index_dir)
        result = s4_taxonomy.run_embedding(
            term_typings=term_typings,
            taxonomy_parent_index=taxonomy_index,
            k=getattr(args, "k", 1),
            threshold=getattr(args, "threshold", 0.0),
            token_overlap_threshold=getattr(args, "token_overlap", 0.0),
        )
    else:
        requested_mode = getattr(args, "s4_mode", "labels")
        if requested_mode == "upperbound":
            if not getattr(args, "s2_gold", None):
                sys.exit("--s2-gold is required for S4 mode 'upperbound'")
            mode = "fewshot_grouped_pos_neg_lexical_child_batches"
            s2_input_path = args.s2_gold
            logger.info(
                "S4 upper-bound: using gold S2 types from %s; strategy=%s",
                s2_input_path,
                mode,
            )
        else:
            mode = requested_mode
            s2_input_path = getattr(args, "s2_input", None)

        fewshot_modes = {
            "fewshot_pos",
            "fewshot_pos_neg",
            "fewshot_grouped_pos_neg",
            "fewshot_grouped_pos_neg_child_batches",
            "fewshot_grouped_pos_neg_lexical",
            "fewshot_grouped_pos_neg_lexical_inverse_pruned",
            "fewshot_grouped_pos_neg_lexical_child_batches",
            "fewshot_grouped_pos_neg_lexical_child_batches_single_parent",
        }

        documents = None
        doc_index = None
        if mode in fewshot_modes:
            import s0_retriever

            if not getattr(args, "corpus", None):
                sys.exit(f"--corpus is required for S4 mode {mode!r}")
            if not getattr(args, "index_dir", None):
                sys.exit(f"--index-dir is required for S4 mode {mode!r} (built by s0)")
            documents = _load_corpus(args.corpus)
            doc_index = s0_retriever.load_doc_index(args.index_dir)
            logger.info(
                "S4 LLM mode=%s: %d corpus docs, k_examples=%d",
                mode,
                len(documents),
                args.k_examples,
            )
        else:
            logger.info("S4 LLM mode=%s", mode)

        common_kwargs = {
            "client": _make_client(args),
            "batch_size": args.batch_size,
            "workers": args.workers,
            "depth_pass": depth_pass,
            "cluster_types": args.cluster_types,
            "n_clusters": args.n_clusters,
            "mode": mode,
            "doc_index": doc_index,
            "documents": documents,
            "k_examples": args.k_examples,
            "max_positive_examples": args.max_positive_examples,
            "max_negative_examples": args.max_negative_examples,
            "example_seed": args.example_seed,
        }

        if s2_input_path:
            types = _load_types(s2_input_path)
            logger.info("S4 LLM: %d S2 types from %s", len(types), s2_input_path)
            result = s4_taxonomy.run_from_types(types, **common_kwargs)
        else:
            if not args.input:
                sys.exit("--s2-input is required for S4 LLM (or use --input for legacy S3 input)")
            term_typings = _load_term_typings(args.input)
            logger.info("S4 LLM: %d term typings (legacy S3 input)", len(term_typings))
            result = s4_taxonomy.run(term_typings, **common_kwargs)

    data = {"taxonomic_relations": [{"parent": r.parent, "child": r.child, "source_doc_ids": r.source_doc_ids} for r in result]}
    _write(args.output, data)

    print(f"\n=== S4 ===")
    print(f"Is-a relations: {len(result)}")
    for r in result[:20]:
        print(f"  {r.child!r:35s} is-a {r.parent!r}")
    if len(result) > 20:
        print(f"  ... ({len(result) - 20} more)")


def run_s5(args: argparse.Namespace) -> None:
    import s5_relations

    if getattr(args, "toy", False):
        documents = TOY_DOCUMENTS
    else:
        if not args.corpus:
            sys.exit("--corpus required for s5 (JSONL corpus file)")
        documents = _load_corpus(args.corpus)
    logger.info("S5: %d documents", len(documents))

    terms = _load_terms(args.s1_input)
    types = _load_types(args.s2_input) if args.s2_input else None
    term_typings = _load_term_typings(args.input)
    logger.info("S5: %d terms, %d term typings", len(terms), len(term_typings))

    # Optional few-shot: gold S5 triples of similar training docs.
    retriever = None
    example_pool: dict | None = None
    k_examples = int(getattr(args, "k_examples", 0))
    if k_examples > 0:
        import s0_retriever
        retriever = s0_retriever.load_doc_index(args.index_dir)
        s2_examples_gold = args.examples_gold.replace("s5_gold", "s2_gold")
        if getattr(args, "full_pool", False):
            # Neighbourhood-faithful: pool = ALL train docs (empty triples for the ~93%
            # relation-free), so each query sees demos matching its neighbourhood.
            example_pool = s5_relations.load_full_example_pool(s2_examples_gold, args.examples_gold)
            logger.info("S5 few-shot: FULL pool (neighbourhood-faithful), k=%d, pool=%d all-train docs",
                        k_examples, len(example_pool))
        else:
            example_pool = s5_relations.load_example_pool(args.examples_gold, s2_examples_gold)
            logger.info("S5 few-shot: k=%d docs, pool=%d relation-bearing docs (%s)",
                        k_examples, len(example_pool), args.examples_gold)

    result = s5_relations.run(
        documents=documents,
        terms=terms,
        term_typings=term_typings,
        client=_make_client(args),
        types=types,
        domain_inference=not args.no_domain_inference,
        workers=args.workers,
        retriever=retriever,
        example_pool=example_pool,
        k_examples=k_examples,
        examples_with_text=not getattr(args, "no_examples_text", False),
        examples_by_types=(getattr(args, "retriever_mode", "text") == "types"),
        n_neg_examples=int(getattr(args, "neg_examples", 0)),
        snap=getattr(args, "snap", False),
        relation_classifier_path=getattr(args, "relation_classifier", None),
        relation_gate_threshold=getattr(args, "gate_threshold", None),
        prompt_v2=getattr(args, "prompt_v2", False),
    )

    data = {
        "non_taxonomic_relations": [
            {"head": r.head, "relation": r.relation, "tail": r.tail, "source_doc_id": r.source_doc_id}
            for r in result
        ]
    }
    _write(args.output, data)

    print(f"\n=== S5 ===")
    print(f"Non-taxonomic relations: {len(result)}")
    for r in result[:20]:
        print(f"  {r.head!r:25s} --[{r.relation}]--> {r.tail!r}")
    if len(result) > 20:
        print(f"  ... ({len(result) - 20} more)")


def run_assemble(args: argparse.Namespace) -> None:
    """Assemble challenge-format submission from per-step intermediate files."""
    corpus = _load_corpus(args.corpus)
    doc_ids = [d["id"] for d in corpus]

    terms = _load_terms(args.s1_input)
    term_typings = _load_term_typings(args.s3_input)
    taxonomic = _load_taxonomy(args.s4_input)
    non_taxonomic = _load_nontaxonomy(args.s5_input)

    output = PipelineOutput(
        term_typings=term_typings,
        taxonomic_relations=taxonomic,
        non_taxonomic_relations=non_taxonomic,
    )
    result = output.to_challenge_format(terms, doc_ids)
    _write(args.output, result)

    total = sum(len(d["primitive-ontology-triples"]) for d in result)
    print(f"\n=== Submission ===")
    print(f"Documents : {len(result)}")
    print(f"Triples   : {total}")
    docs_with_triples = sum(1 for d in result if d["primitive-ontology-triples"])
    print(f"Non-empty : {docs_with_triples}/{len(result)} documents")


def run_eval(args: argparse.Namespace) -> None:
    from evaluate import (
        eval_s1_per_doc, eval_s2_per_doc, eval_s3_per_doc,
        eval_s4_per_doc, eval_s5_per_doc, eval_submission_per_doc,
        graph_similarity, print_graph_similarity, print_summary,
    )

    with open(args.gold) as f:
        gold = json.load(f)
    with open(args.predictions) as f:
        predicted = json.load(f)

    print(f"\n=== Evaluation: step={args.step} ===")
    print(f"  Predictions : {args.predictions}")
    print(f"  Gold        : {args.gold}")

    eval_fn = {
        "s1": eval_s1_per_doc,
        "s2": eval_s2_per_doc,
        "s3": eval_s3_per_doc,
        "s4": eval_s4_per_doc,
        "s5": eval_s5_per_doc,
        "submission": eval_submission_per_doc,
    }[args.step]
    labels = {
        "s1": "S1 — Terms",
        "s2": "S2 — Types",
        "s3": "S3 — Term typings (term, type) pairs",
        "s4": "S4 — Taxonomy (parent, child) pairs",
        "s5": "S5 — Non-taxonomic (head, relation, tail)",
        "submission": "Submission — primitive-ontology-triples",
    }

    per_doc = eval_fn(predicted, gold)

    # Save one file per step with all documents' details inside
    eval_path = Path(args.predictions).parent / f"eval_{args.step}.json"
    eval_path.write_text(json.dumps(per_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Detailed eval → {eval_path}")

    # Log only averages
    print_summary(per_doc, label=labels[args.step])

    # For submission: also compute the official graph similarity (per-doc averaged)
    if args.step == "submission":
        modes = ("exact", "fuzzy") if getattr(args, "no_semantic", False) else ("exact", "fuzzy", "semantic")
        try:
            gs = graph_similarity(predicted, gold, modes=modes)
            print_graph_similarity(gs, label="Graph Similarity (official organizer metric, per-doc avg)")
        except Exception as e:
            print(f"  [Graph Similarity skipped: {e}]  (semantic needs the nomic-embed model; try --no-semantic)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one pipeline step or evaluate a step output against gold.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- s0: build embedding retriever index + optional embedding doc index ---
    p0 = sub.add_parser("s0", help="Build retriever indices for S3 RAG and S1S2 few-shot (no LLM)")
    p0.add_argument("--scope", choices=["whole", "split"], default=_SCOPE,
                    help="Config scope: whole or split (picks index/data paths automatically)")
    p0.add_argument("--gold", default="data/gold/s3_gold.json",
                    help="Path to s3_gold.json for the pair index (default: data/gold/s3_gold.json)")
    p0.add_argument("--index-dir", default=_CFG.get("index_dir"),
                    help="Directory to write the retriever index (default: config index_dir)")
    p0.add_argument("--embedding-model", default=_CFG.get("embedding_model", "all-MiniLM-L6-v2"),
                    help="SentenceTransformer model name (default: config embedding_model)")
    p0.add_argument("--batch-size", type=int, default=512,
                    help="Encoding batch size (default: 512)")
    p0.add_argument("--training-data", default=_CFG.get("training_data"),
                    help="Path to train_task_a.json — also builds embedding doc index (default: config training_data)")
    p0.add_argument("--correct-terms", action="store_true",
                    help="Apply gold term correction (text-form matching) when building doc index")

    # --- s1s2: separate per-document term extraction (S1) then type extraction (S2) ---
    p12 = sub.add_parser("s1s2", help="Per-document term extraction (S1) then type extraction (S2) with RAG few-shot")
    _common_args(p12, require_output=False)
    p12.add_argument("--input", default=None, help="JSONL corpus (one {id,text} per line)")
    p12.add_argument("--toy", action="store_true", help="Use built-in smart-home toy document")
    p12.add_argument("--index-dir", default=None,
                     help="S0 index directory (for embedding doc few-shot); omit for zero-shot")
    p12.add_argument("--s1-output", required=True, help="Output path for S1 terms JSON")
    p12.add_argument("--s2-output", required=True, help="Output path for S2 types JSON")
    p12.add_argument("--k-examples-s1", type=int, default=int(_CFG.get("k_examples_s1", 3)),
                     help="Training docs retrieved per document for S1 RAG few-shot (default: 3)")
    p12.add_argument("--k-examples-s2", type=int, default=int(_CFG.get("k_examples_s2", 3)),
                     help="Training docs retrieved per document for S2 RAG few-shot (default: 3)")
    p12.add_argument("--retriever-mode", default="text", choices=["text", "terms"],
                     help="RAG retrieval mode: 'text' (full doc) or 'terms' (gold terms) (default: text)")
    p12.add_argument("--example-selection-s1", default="similarity_all_docs",
                     choices=["similarity_all_docs", "similarity_two_pools", "random"],
                     help="S1 example selection mode (default: similarity_all_docs)")
    p12.add_argument("--k-empty-s1", type=int, default=0,
                     help="S1 dedicated type-only examples (used with similarity_two_pools, default: 0)")
    p12.add_argument("--classifier", default=None,
                     help="Path to term classifier pkl — skips LLM for predicted type-only docs")

    # --- s1: standalone term extraction ---
    p1 = sub.add_parser("s1", help="Term extraction from corpus passages")
    _common_args(p1)
    p1.add_argument("--input", default=None, help="JSONL corpus (one {id,text} per line)")
    p1.add_argument("--toy", action="store_true", help="Use built-in smart-home toy document")
    p1.add_argument("--index-dir", default=_CFG.get("index_dir_s1") or _CFG.get("index_dir"),
                    help="S0 index directory (default: config index_dir_s1 or index_dir; "
                         "set index_dir_s1 to the corrected index for the terms+corrected combo)")
    p1.add_argument("--k-examples", type=int, default=int(_CFG.get("k_examples_s1", 15)))
    p1.add_argument("--retriever-mode", default=_CFG.get("retriever_mode_s1", "text"),
                    choices=["text", "terms"],
                    help="RAG retrieval mode: 'text' or 'terms' (default: config retriever_mode_s1)")
    p1.add_argument("--cot", action="store_true",
                    default=str(_CFG.get("cot_s1", "false")).lower() == "true",
                    help="Use Chain-of-Thought prompt (default from config cot_s1; best = off)")
    p1.add_argument("--example-selection",
                    default=_CFG.get("example_selection_s1", "similarity_all_docs"),
                    choices=["similarity_all_docs", "similarity_two_pools", "random"],
                    help="S1 example selection mode (default: config example_selection_s1)")
    p1.add_argument("--k-empty", type=int, default=int(_CFG.get("k_empty_s1", 0)),
                    help="S1 dedicated type-only examples (used with similarity_two_pools)")
    p1.add_argument("--classifier", default=_CFG.get("classifier_s1"),
                    help="Path to term classifier pkl — skips LLM for predicted type-only docs")

    # --- s2: type vocabulary (gold pass-through, no LLM) ---
    p2 = sub.add_parser("s2", help="Per-document type extraction with RAG few-shot")
    _common_args(p2)
    p2.add_argument("--input", default=None, help="JSONL corpus (one {id,text} per line)")
    p2.add_argument("--toy", action="store_true", help="Use built-in smart-home toy document")
    p2.add_argument("--index-dir", default=_CFG.get("index_dir_s2") or _CFG.get("index_dir"),
                    help="S0 index directory for RAG few-shot (default: config index_dir_s2 or index_dir)")
    p2.add_argument("--k-examples", type=int, default=int(_CFG.get("k_examples_s2", 20)),
                    help="Few-shot examples per doc (default: 20). NOTE: run S2 with --no-thinking")
    p2.add_argument("--cot", action="store_true",
                    default=str(_CFG.get("cot_s2", "false")).lower() == "true",
                    help="Use Chain-of-Thought prompt (default from config cot_s2)")
    p2.add_argument("--retriever-mode", default=_CFG.get("retriever_mode_s2", "text"),
                    choices=["text", "terms"],
                    help="RAG retrieval mode: 'text' or 'types' (default: config retriever_mode_s2)")
    p2.add_argument("--s2-example-selection",
                    choices=sorted(s2_type_extraction.EXAMPLE_SELECTION_MODES),
                    default=_CFG.get("s2_example_selection", "similarity"),
                    help="S2 few-shot selection mode (default: config s2_example_selection)")
    p2.add_argument("--s2-example-seed", type=int,
                    default=(int(_CFG["s2_example_seed"]) if _CFG.get("s2_example_seed") else None),
                    help="Seed for deterministic S2 random few-shot selection (default: config)")

    # --- s3: term typing (RAG-based) ---
    p3 = sub.add_parser("s3", help="Term typing — RAG-based (reads s1 output + s0 index)")
    _common_args(p3)
    p3.add_argument("--s1-input", required=True, help="S1 output JSON file (terms)")
    p3.add_argument("--s2-input", default=None,
                    help="S2 output JSON file (types); enables vocab snapping in S3")
    p3.add_argument("--index-dir", default=_CFG.get("index_dir_s3") or _CFG.get("index_dir"),
                    help="S0 retriever index directory (default: config index_dir_s3 or index_dir)")
    p3.add_argument("--corpus", default=None,
                    help="JSONL corpus (one {id,text} per line) — required for --context != none (full-doc grounding)")
    p3.add_argument("--context", default=_CFG.get("context_s3", "full"),
                    choices=["full", "sentence", "title_sentence", "none"],
                    help="Grounding context per term (default: full = best)")
    p3.add_argument("--no-snap", action="store_true",
                    default=str(_CFG.get("snap_s3", "true")).lower() == "false",
                    help="Disable snapping OOV predicted types to the nearest S2 type "
                         "(snapping is on by default and needs --s2-input)")
    p3.add_argument("--k", type=int, default=int(_CFG.get("k_s3", 10)),
                    help="Nearest-neighbour examples per term (default: 10)")
    p3.add_argument("--batch-size", type=int, default=int(_CFG.get("batch_size_s3", 0)),
                    help="Terms per LLM call; 0 = one call per doc (default: 0)")
    # Alternative typing modes (off by default; mutually exclusive with free generation)
    p3.add_argument("--constrained", action="store_true",
                    default=str(_CFG.get("constrained_s3", "false")).lower() == "true",
                    help="Closed-world: pick types only from the per-term retrieved candidate list")
    p3.add_argument("--lenient", action="store_true",
                    default=str(_CFG.get("lenient_s3", "false")).lower() == "true",
                    help="Candidates as suggestions with a free-generation escape hatch")
    p3.add_argument("--augment-frequent", type=int, default=int(_CFG.get("augment_frequent_s3", 0)),
                    help="Show top-N most-frequent train types as a prompt prior (0 = off)")
    p3.add_argument("--hybrid", action="store_true",
                    default=str(_CFG.get("hybrid_s3", "false")).lower() == "true",
                    help="Hybrid retrieval: embedding cosine + lexical term-match boost")
    p3.add_argument("--hybrid-alpha", type=float, default=float(_CFG.get("hybrid_alpha_s3", 0.5)),
                    help="Lexical weight in hybrid retrieval (default: 0.5)")

    # --- s4: taxonomy discovery ---
    p4 = sub.add_parser(
        "s4",
        help="Taxonomy discovery (LLM reads S2 types; embedding reads S3 typings)",
    )
    _common_args(p4)
    p4.add_argument(
        "--input",
        default=None,
        help="S3 output JSON file (required for --embedding; legacy fallback for LLM)",
    )
    p4.add_argument(
        "--s2-input",
        default=None,
        help="S2 output JSON file (types); used by the LLM approach",
    )
    p4.add_argument(
        "--s2-gold",
        default=None,
        help="Gold S2 JSON file; required by --s4-mode upperbound",
    )
    p4.add_argument(
        "--corpus",
        default=None,
        help="Corpus JSONL; required for S4 few-shot modes",
    )
    p4.add_argument("--batch-size", type=int, default=int(_CFG.get("batch_size_s4", 10)))
    p4.add_argument(
        "--s4-mode",
        choices=[
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
            "upperbound",
        ],
        default=_CFG.get("s4_mode", "labels"),
        help="S4 LLM mode or input experiment (default: config s4_mode)",
    )
    p4.add_argument(
        "--k-examples",
        type=int,
        default=int(_CFG.get("k_examples_s4", 3)),
        help="Similar training docs for S4 few-shot modes",
    )
    p4.add_argument(
        "--max-positive-examples",
        type=int,
        default=int(_CFG.get("s4_max_positive_examples", 20)),
        help="Max positive is-a examples per S4 few-shot prompt",
    )
    p4.add_argument(
        "--max-negative-examples",
        type=int,
        default=int(_CFG.get("s4_max_negative_examples", 20)),
        help="Max generated negative examples per S4 few-shot prompt",
    )
    p4.add_argument(
        "--example-seed",
        type=int,
        default=_cfg_optional_int("s4_example_seed"),
        help="Optional seed for deterministic S4 negative examples",
    )
    depth_group = p4.add_mutually_exclusive_group()
    depth_group.add_argument(
        "--depth-pass",
        action="store_true",
        help="Run the depth re-prompt second pass",
    )
    depth_group.add_argument(
        "--no-depth-pass",
        action="store_true",
        help="Skip the depth re-prompt second pass",
    )
    p4.add_argument("--cluster-types", action="store_true",
                    help="Enable semantic clustering pre-step")
    p4.add_argument("--n-clusters", type=int, default=int(_CFG.get("n_clusters_s4", 10)),
                    help="Target cluster count when --cluster-types is set (default: 10)")
    p4.add_argument("--embedding", action="store_true",
                    help="Use embedding similarity approach instead of LLM; requires --index-dir")
    p4.add_argument("--index-dir", default=_CFG.get("index_dir_s4") or _CFG.get("index_dir"),
                    help="S0 index directory for S4 few-shot doc retrieval (default: config index_dir)")
    p4.add_argument("--k", type=int, default=1,
                    help="Number of nearest training parents per test type when --embedding is set (default: 1)")
    p4.add_argument("--threshold", type=float, default=float(_CFG.get("s4_threshold", 0.0)),
                    help="Minimum cosine similarity to emit a pair when --embedding is set (default: 0.0)")
    p4.add_argument("--token-overlap", type=float, default=float(_CFG.get("s4_token_overlap", 0.0)),
                    help="Minimum token overlap F1 between child/parent labels (default: 0.0 = disabled)")

    # --- s5: non-taxonomic relation extraction ---
    p5 = sub.add_parser("s5", help="Non-taxonomic relation extraction (open RE)")
    _common_args(p5)
    p5.add_argument("--corpus", default=None,
                    help="Original JSONL corpus (one {id,text} per line)")
    p5.add_argument("--toy", action="store_true", help="Use built-in smart-home toy document")
    p5.add_argument("--s1-input", required=True, help="S1 output JSON file (terms)")
    p5.add_argument("--s2-input", default=None, help="S2 output JSON file (types, optional)")
    p5.add_argument("--input", required=True, help="S3 output JSON file (term typings)")
    p5.add_argument("--no-domain-inference", action="store_true",
                    default=str(_CFG.get("domain_inference_s5", "true")).lower() == "false",
                    help="Disable domain inference pre-step (P1) (default from config domain_inference_s5)")
    p5.add_argument("--k-examples", type=int, default=int(_CFG.get("k_examples_s5", 0)),
                    help="Few-shot: k nearest training docs as types→relations examples (0 = off)")
    p5.add_argument("--no-examples-text", action="store_true",
                    default=str(_CFG.get("s5_examples_with_text", "true")).lower() == "false",
                    help="Exclude example docs' passages from the few-shot block (types→relations only)")
    p5.add_argument("--retriever-mode", default=_CFG.get("retriever_mode_s5", "text"),
                    choices=["text", "types"],
                    help="Few-shot retrieval similarity: doc text or S2 type-vocabulary overlap")
    p5.add_argument("--neg-examples", type=int, default=int(_CFG.get("neg_examples_s5", 0)),
                    help="Include N nearest relation-free docs as negative demos (types→none)")
    p5.add_argument("--index-dir", default=_CFG.get("index_dir_s5") or _CFG.get("index_dir"),
                    help="Doc embedding index for few-shot retrieval")
    p5.add_argument("--examples-gold", default=_CFG.get("s5_examples_gold", "data/gold/s5_gold.json"),
                    help="Training S5 gold to retrieve few-shot triples from")
    p5.add_argument("--snap", action="store_true",
                    default=str(_CFG.get("snap_s5", "false")).lower() == "true",
                    help="Snap predicted head/tail to the doc's S2 type vocab (needs --s2-input)")
    p5.add_argument("--relation-classifier", default=_CFG.get("relation_classifier_s5") or None,
                    help="Pkl gate — skip LLM for docs predicted relation-free (empty-doc gate)")
    _gt = _CFG.get("gate_threshold_s5", "")
    p5.add_argument("--gate-threshold", type=float, default=(float(_gt) if _gt else None),
                    help="Override gate decision threshold (higher = filter more docs; macro-F1 favours this)")
    p5.add_argument("--full-pool", action="store_true",
                    default=str(_CFG.get("full_pool_s5", "false")).lower() == "true",
                    help="Neighbourhood-faithful few-shot: draw the k nearest from ALL train docs "
                         "(empty demos for relation-free) — handles empty docs without a gate")
    p5.add_argument("--prompt-v2", action="store_true",
                    default=str(_CFG.get("prompt_v2_s5", "false")).lower() == "true",
                    help="Use the revised S5 prompt (no is-a rule, use-context, synonym nudge, disjoint guard)")

    # --- assemble: build submission from intermediate files ---
    pa = sub.add_parser("assemble", help="Assemble challenge submission from step outputs (no LLM)")
    pa.add_argument("--corpus", required=True, help="Original JSONL corpus (preserves doc order)")
    pa.add_argument("--s1-input", required=True, help="S1 output JSON (terms)")
    pa.add_argument("--s3-input", required=True, help="S3 output JSON (term typings)")
    pa.add_argument("--s4-input", required=True, help="S4 output JSON (taxonomic relations)")
    pa.add_argument("--s5-input", required=True, help="S5 output JSON (non-taxonomic relations)")
    pa.add_argument("--output", required=True, help="Submission JSON output file")

    # --- eval ---
    pe = sub.add_parser("eval", help="Evaluate a step output against the per-step gold")
    pe.add_argument("--step", required=True,
                    choices=["s1", "s2", "s3", "s4", "s5", "submission"],
                    help="Step whose output is being evaluated")
    pe.add_argument("--predictions", required=True, help="Step output JSON file")
    pe.add_argument("--gold", required=True,
                    help="Gold JSON file (e.g. data/gold/s3_gold.json)")
    pe.add_argument("--no-semantic", action="store_true",
                    help="Submission graph similarity: skip the semantic mode "
                         "(avoids the nomic-embed model download; exact+fuzzy only)")

    args = parser.parse_args()

    dispatch = {
        "s0": run_s0,
        "s1s2": run_s1s2,
        "s1": run_s1,
        "s2": run_s2,
        "s3": run_s3,
        "s4": run_s4,
        "s5": run_s5,
        "assemble": run_assemble,
        "eval": run_eval,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
