"""Data models for the ontology learning pipeline."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedTerm:
    """A term extracted from one or more corpus documents (S1 output)."""

    text: str
    source_doc_ids: list[str] = field(default_factory=list)
    context_sentence: str = ""


@dataclass
class ExtractedType:
    """A semantic type (class label) extracted from the corpus (S2 output)."""

    text: str
    source_doc_ids: list[str] = field(default_factory=list)


@dataclass
class TermTypingResult:
    """A term paired with its predicted semantic types (S3 output)."""

    term: str
    types: list[str]
    source_doc_ids: list[str] = field(default_factory=list)


@dataclass
class TaxonomyResult:
    """A single is-a relationship between two concepts (S4 output)."""

    parent: str
    child: str
    source_doc_ids: list[str] = field(default_factory=list)


@dataclass
class RelationResult:
    """A non-hierarchical semantic relation between two concepts (S5 output)."""

    head: str
    relation: str
    tail: str
    source_doc_id: str = ""


@dataclass
class PipelineOutput:
    """Aggregated output of the complete ontology learning pipeline."""

    term_typings: list[TermTypingResult]
    taxonomic_relations: list[TaxonomyResult]
    non_taxonomic_relations: list[RelationResult]

    def to_challenge_format(
        self,
        terms: list[ExtractedTerm],
        doc_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Serialize to the LLMs4OL challenge submission format.

        Each input document gets one entry with the triples derived from it.
        Triples are assigned to documents via the term → source_doc_ids mapping
        from S1: is-a triples follow the child term's docs, non-taxonomic
        triples follow the head term's docs.

        Args:
            terms: S1 output carrying term → source_doc_ids links.
            doc_ids: Ordered list of document IDs (preserves submission order).

        Returns:
            List of ``{"id": doc_id, "primitive-ontology-triples": [[s, r, o], ...]}``
            one entry per document in ``doc_ids``.
        """
        term_to_docs: dict[str, list[str]] = {
            t.text: t.source_doc_ids for t in terms
        }

        doc_triples: dict[str, dict[tuple[str, str, str], None]] = defaultdict(dict)

        # S3 — instance-of triples: doc_ids come from the TermTypingResult itself
        for tt in self.term_typings:
            for typ in tt.types:
                for doc_id in tt.source_doc_ids:
                    doc_triples[doc_id][(tt.term, "instance-of", typ)] = None

        # S4 — is-a triples: doc_ids come from the TaxonomyResult itself
        for tr in self.taxonomic_relations:
            for doc_id in tr.source_doc_ids:
                doc_triples[doc_id][(tr.child, "is-a", tr.parent)] = None

        # S5 — non-taxonomic triples; use source_doc_id when available
        for rel in self.non_taxonomic_relations:
            if rel.source_doc_id:
                doc_triples[rel.source_doc_id][(rel.head, rel.relation, rel.tail)] = None
            else:
                for doc_id in term_to_docs.get(rel.head, []):
                    doc_triples[doc_id][(rel.head, rel.relation, rel.tail)] = None

        return [
            {
                "id": doc_id,
                "primitive-ontology-triples": [list(t) for t in doc_triples.get(doc_id, {})],
            }
            for doc_id in doc_ids
        ]

    def to_ontolearner(self) -> "OntologyData":  # noqa: F821
        """Convert to OntoLearner's ``OntologyData`` Pydantic model.

        Raises:
            ImportError: If ``ontolearner`` is not installed.
        """
        from ontolearner.data_structure.data import (  # type: ignore[import]
            NonTaxonomicRelation,
            NonTaxonomicRelations,
            OntologyData,
            TaxonomicRelation,
            TermTyping,
            TypeTaxonomies,
        )

        ol_term_typings = [
            TermTyping(term=tt.term, types=tt.types)
            for tt in self.term_typings
            if tt.types
        ]

        taxonomy_types: list[str] = list(
            {c for tr in self.taxonomic_relations for c in (tr.parent, tr.child)}
        )
        ol_taxonomies = [
            TaxonomicRelation(parent=tr.parent, child=tr.child)
            for tr in self.taxonomic_relations
        ]

        nt_types: list[str] = list(
            {c for r in self.non_taxonomic_relations for c in (r.head, r.tail)}
        )
        nt_relation_types: list[str] = list(
            {r.relation for r in self.non_taxonomic_relations}
        )
        ol_non_taxonomies = [
            NonTaxonomicRelation(head=r.head, tail=r.tail, relation=r.relation)
            for r in self.non_taxonomic_relations
        ]

        return OntologyData(
            term_typings=ol_term_typings,
            type_taxonomies=TypeTaxonomies(
                types=taxonomy_types,
                taxonomies=ol_taxonomies,
            ),
            type_non_taxonomic_relations=NonTaxonomicRelations(
                types=nt_types,
                relations=nt_relation_types,
                non_taxonomies=ol_non_taxonomies,
            ),
        )
