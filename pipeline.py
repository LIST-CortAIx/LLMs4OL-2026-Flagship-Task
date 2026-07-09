"""End-to-end ontology learning pipeline.

Chains the five steps:
  S1 — Term Extraction
  S2 — Type Extraction
  S3 — Term Typing
  S4 — Taxonomy Discovery
  S5 — Non-Taxonomic Relation Extraction

Entry point: ``Pipeline.run(documents)`` → ``PipelineOutput``

The pipeline can also be run step-by-step for debugging or incremental
execution (useful when iterating on a single step without re-running all
prior steps).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from llm import LLMClient
from models import (
    ExtractedTerm,
    ExtractedType,
    PipelineOutput,
    TermTypingResult,
    TaxonomyResult,
    RelationResult,
)
import s0_retriever
import s1_term_extraction
import s2_type_extraction
import s3_term_typing
import s4_taxonomy
import s5_relations

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for a full pipeline run.

    Args:
        k_examples: Training docs retrieved per document for RAG few-shot in S1 and S2.
        batch_size_s3: Terms per LLM call in S3.
        batch_size_s4: Candidate pairs per LLM call in S4.
        workers: Thread-pool size for parallel LLM calls (shared across steps).
        max_consecutive_errors: Abort threshold for consecutive LLM failures.
        index_dir: Directory containing the S0 retriever index (S3 RAG + S1/S2 doc index).
        k_s3: Number of nearest-neighbour examples retrieved per term in S3.
        s4_depth_pass: Whether to run the depth re-prompt second pass in S4.
        s4_cluster_types: Whether to cluster types before generating pairs in S4.
        s4_n_clusters: Target number of clusters for S4 clustering.
        s4_mode: S4 LLM prompting mode.
        k_examples_s4: Similar training documents for S4 few-shot modes.
        s4_max_positive_examples: Maximum positive S4 examples per document.
        s4_max_negative_examples: Maximum generated negative S4 examples per document.
        s4_example_seed: Optional deterministic seed for S4 negative examples.
        s5_domain_inference: Whether to run the domain inference pre-step in S5.
        examples_s5: Few-shot examples for S5 prompts.
    """

    k_examples: int = 3
    batch_size_s3: int = 10
    batch_size_s4: int = 20
    workers: int = 4
    max_consecutive_errors: int = 5
    index_dir: str = "outputs/s0_retriever"
    k_s3: int = 3
    s4_depth_pass: bool = False
    s4_cluster_types: bool = False
    s4_n_clusters: int = 10
    s4_mode: str = "labels"
    k_examples_s4: int = 3
    s4_max_positive_examples: int = 20
    s4_max_negative_examples: int = 20
    s4_example_seed: int | None = None
    s5_domain_inference: bool = True
    examples_s5: list[dict[str, Any]] = field(default_factory=list)


class Pipeline:
    """Full ontology learning pipeline backed by a vLLM server.

    Args:
        client: ``LLMClient`` pointing at a running vLLM server.
        config: ``PipelineConfig`` with step-level settings.
    """

    def __init__(
        self,
        client: LLMClient,
        config: PipelineConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or PipelineConfig()

    def run(
        self,
        documents: list[dict[str, str]],
    ) -> PipelineOutput:
        """Run the full pipeline on a corpus of documents.

        Args:
            documents: List of ``{"id": ..., "text": ...}`` dicts. Each
                document must have a unique ``id`` and non-empty ``text``.

        Returns:
            ``PipelineOutput`` with term typings, taxonomic and
            non-taxonomic relations, convertible to OntoLearner format via
            ``output.to_ontolearner()``.
        """
        t0 = time.perf_counter()
        logger.info("=== Pipeline start: %d documents ===", len(documents))

        terms = self.run_s1(documents)
        types = self.run_s2(documents)
        term_typings = self.run_s3(terms, types)
        taxonomic = self.run_s4(
            term_typings,
            types=types,
            documents=documents,
        )
        non_taxonomic = self.run_s5(term_typings, documents=documents, terms=terms, types=types)

        elapsed = time.perf_counter() - t0
        logger.info(
            "=== Pipeline done in %.1fs: %d term-typings, %d is-a, %d relations ===",
            elapsed, len(term_typings), len(taxonomic), len(non_taxonomic),
        )
        return PipelineOutput(
            term_typings=term_typings,
            taxonomic_relations=taxonomic,
            non_taxonomic_relations=non_taxonomic,
        )

    # ------------------------------------------------------------------
    # Individual step runners (useful for step-by-step execution)
    # ------------------------------------------------------------------

    def _load_doc_index(self):
        """Load the doc embedding index for RAG few-shot (S1/S2)."""
        from pathlib import Path
        import s0_retriever
        idx_path = Path(self.config.index_dir) / "doc_embeddings.npy"
        if idx_path.exists():
            return s0_retriever.load_doc_index(self.config.index_dir)
        return None

    def run_s1(self, documents: list[dict[str, str]]) -> list[ExtractedTerm]:
        """Run Step 1: per-document term extraction with RAG few-shot."""
        cfg = self.config
        return s1_term_extraction.run(
            documents=documents,
            client=self.client,
            workers=cfg.workers,
            doc_index=self._load_doc_index(),
            k_examples=cfg.k_examples,
            max_consecutive_errors=cfg.max_consecutive_errors,
        )

    def run_s2(self, documents: list[dict[str, str]]) -> list[ExtractedType]:
        """Run Step 2: per-document type extraction with RAG few-shot."""
        cfg = self.config
        return s2_type_extraction.run(
            documents=documents,
            client=self.client,
            workers=cfg.workers,
            doc_index=self._load_doc_index(),
            k_examples=cfg.k_examples,
            max_consecutive_errors=cfg.max_consecutive_errors,
        )

    def run_s3(
        self,
        terms: list[ExtractedTerm],
        types: list[ExtractedType],  # noqa: ARG002
    ) -> list[TermTypingResult]:
        """Run Step 3: RAG-based term typing.

        The ``types`` argument is accepted for API compatibility but is not
        used — S3 derives types from retrieved training examples, not from
        the closed S2 vocabulary.
        """
        import s0_retriever

        cfg = self.config
        retriever = s0_retriever.load(cfg.index_dir)
        return s3_term_typing.run(
            terms=terms,
            client=self.client,
            retriever=retriever,
            k=cfg.k_s3,
            batch_size=cfg.batch_size_s3,
            workers=cfg.workers,
            max_consecutive_errors=cfg.max_consecutive_errors,
        )

    def run_s4(
        self,
        term_typings: list[TermTypingResult],
        types: list[ExtractedType] | None = None,
        documents: list[dict[str, str]] | None = None,
    ) -> list[TaxonomyResult]:
        """Run Step 4: taxonomy discovery."""
        cfg = self.config
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
        doc_index = (
            s0_retriever.load_doc_index(cfg.index_dir)
            if cfg.s4_mode in fewshot_modes
            else None
        )
        kwargs = {
            "client": self.client,
            "batch_size": cfg.batch_size_s4,
            "workers": cfg.workers,
            "max_consecutive_errors": cfg.max_consecutive_errors,
            "depth_pass": cfg.s4_depth_pass,
            "cluster_types": cfg.s4_cluster_types,
            "n_clusters": cfg.s4_n_clusters,
            "mode": cfg.s4_mode,
            "doc_index": doc_index,
            "documents": documents,
            "k_examples": cfg.k_examples_s4,
            "max_positive_examples": cfg.s4_max_positive_examples,
            "max_negative_examples": cfg.s4_max_negative_examples,
            "example_seed": cfg.s4_example_seed,
        }
        if types is not None:
            return s4_taxonomy.run_from_types(types=types, **kwargs)
        return s4_taxonomy.run(term_typings=term_typings, **kwargs)

    def run_s5(
        self,
        term_typings: list[TermTypingResult],
        documents: list[dict[str, str]] | None = None,
        terms: list[ExtractedTerm] | None = None,
        types: list[ExtractedType] | None = None,
    ) -> list[RelationResult]:
        """Run Step 5: non-taxonomic relation extraction (text + semantic inference)."""
        cfg = self.config
        return s5_relations.run(
            documents=documents or [],
            terms=terms or [],
            term_typings=term_typings,
            client=self.client,
            types=types,
            domain_inference=cfg.s5_domain_inference,
            workers=cfg.workers,
            max_consecutive_errors=cfg.max_consecutive_errors,
            examples=cfg.examples_s5 or None,
        )


def _load_config() -> dict[str, str]:
    """Read key: value pairs from config.yaml at the project root."""
    from pathlib import Path
    cfg: dict[str, str] = {}
    config_file = Path(__file__).resolve().parent / "config.yaml"
    if config_file.exists():
        for line in config_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and ":" in line:
                key, _, val = line.partition(":")
                cfg[key.strip()] = val.strip()
    return cfg


def main() -> None:
    """CLI entry point: ``run-pipeline``  (see pyproject.toml scripts)."""
    import argparse
    import json
    import os
    import sys

    _cfg = _load_config()

    parser = argparse.ArgumentParser(
        description="Run the ontology learning pipeline on a JSONL corpus."
    )
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL") or _cfg.get("model"),
                        help="Model path as registered in the vLLM server (default: $LLM_MODEL or config.yaml)")
    parser.add_argument("--base-url",
                        default=os.environ.get("LLM_BASE_URL") or _cfg.get("base_url"),
                        help="vLLM server URL (default: $LLM_BASE_URL or config.yaml)")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Input JSONL file (one doc per line: {id, text})")
    input_group.add_argument("--toy", action="store_true", help="Use built-in smart-home toy document")
    parser.add_argument("--output", required=True, help="Output JSON file for PipelineOutput")
    parser.add_argument("--challenge-format", action="store_true",
                        help="Output primitive-ontology-triples per document (challenge submission format)")
    parser.add_argument("--k-examples", type=int, default=3,
                        help="Training docs retrieved per document for RAG few-shot in S1/S2 (default: 3)")
    parser.add_argument("--batch-size-s3", type=int, default=10)
    parser.add_argument("--batch-size-s4", type=int, default=20)
    parser.add_argument("--index-dir", default="outputs/s0_retriever",
                        help="S0 retriever index directory (default: outputs/s0_retriever)")
    parser.add_argument("--k-s3", type=int, default=3,
                        help="Nearest-neighbour examples per term in S3 (default: 3)")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=float(_cfg.get("temperature", 1.0)))
    parser.add_argument("--top-p", type=float, default=float(_cfg.get("top_p", 0.95)))
    parser.add_argument("--top-k", type=int, default=int(_cfg.get("top_k", 20)))
    parser.add_argument("--min-p", type=float, default=float(_cfg.get("min_p", 0.0)))
    parser.add_argument("--presence-penalty", type=float, default=float(_cfg.get("presence_penalty", 1.5)))
    parser.add_argument("--repetition-penalty", type=float, default=float(_cfg.get("repetition_penalty", 1.0)))
    parser.add_argument("--s4-cluster-types", action="store_true",
                        help="Enable S4 semantic clustering pre-step (P2)")
    parser.add_argument("--s4-n-clusters", type=int, default=10,
                        help="Target number of clusters for S4 clustering (default: 10)")
    parser.add_argument("--no-s5-domain-inference", action="store_true",
                        help="Disable S5 domain inference pre-step (P1)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel HTTP calls to the vLLM server")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if not args.base_url:
        parser.error("--base-url is required (or set $LLM_BASE_URL / config.yaml)")

    if args.toy:
        documents = [{"id": "doc1", "text": (
            "In a smart home system, sensors monitor environmental conditions. "
            "A temperature sensor measures room temperature. "
            "A motion sensor detects movement and triggers the alarm system. "
            "The smart thermostat receives temperature readings from the temperature sensor "
            "and adjusts the heating system. "
            "A mobile app allows the user to control the smart thermostat remotely."
        )}]
    else:
        documents = []
        with open(args.input) as f:
            for line in f:
                line = line.strip()
                if line:
                    documents.append(json.loads(line))

    client = LLMClient(
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        json_mode=True,
    )

    config = PipelineConfig(
        k_examples=args.k_examples,
        batch_size_s3=args.batch_size_s3,
        batch_size_s4=args.batch_size_s4,
        workers=args.workers,
        index_dir=args.index_dir,
        k_s3=args.k_s3,
        s4_cluster_types=args.s4_cluster_types,
        s4_n_clusters=args.s4_n_clusters,
        s5_domain_inference=not args.no_s5_domain_inference,
    )
    pipeline = Pipeline(client=client, config=config)

    if args.challenge_format:
        terms = pipeline.run_s1(documents)
        types = pipeline.run_s2(documents)
        term_typings = pipeline.run_s3(terms, types)
        taxonomic = pipeline.run_s4(
            term_typings,
            types=types,
            documents=documents,
        )
        non_taxonomic = pipeline.run_s5(term_typings, documents=documents, terms=terms, types=types)
        output = PipelineOutput(
            term_typings=term_typings,
            taxonomic_relations=taxonomic,
            non_taxonomic_relations=non_taxonomic,
        )
        doc_ids = [d["id"] for d in documents]
        result = output.to_challenge_format(terms, doc_ids)
    else:
        output = pipeline.run(documents)
        result = {
            "term_typings": [
                {"term": tt.term, "types": tt.types}
                for tt in output.term_typings
            ],
            "taxonomic_relations": [
                {"parent": r.parent, "child": r.child}
                for r in output.taxonomic_relations
            ],
            "non_taxonomic_relations": [
                {"head": r.head, "relation": r.relation, "tail": r.tail}
                for r in output.non_taxonomic_relations
            ],
        }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("Output written to %s", args.output)
