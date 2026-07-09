"""Unit tests for the flat Option 2 submission.

The tests avoid external model calls and focus on deterministic core behavior:
format conversion, JSON parsing, evaluation helpers and file loaders.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ORIGINAL_SYS_PATH = list(sys.path)
sys.path = [
    item for item in sys.path
    if Path(item or ".").resolve() != _SCRIPT_DIR
]
import unittest  # noqa: E402
sys.path = _ORIGINAL_SYS_PATH

from evaluate import eval_submission_per_doc, macro_average, micro_average
from llm import LLMError, _extract_json
from models import ExtractedTerm, PipelineOutput, RelationResult, TaxonomyResult, TermTypingResult
from run_step import _load_corpus, _load_taxonomy, _load_term_typings, _load_terms, _load_types
from s1_term_extraction import _find_context_sentence
from s3_term_typing import _build_contexts, _containing_sentence, _doc_title


class ChallengeFormatTests(unittest.TestCase):
    def test_to_challenge_format_assigns_and_deduplicates_triples(self) -> None:
        output = PipelineOutput(
            term_typings=[
                TermTypingResult("temperature sensor", ["sensor", "sensor"], ["doc1"]),
            ],
            taxonomic_relations=[
                TaxonomyResult(parent="device", child="sensor", source_doc_ids=["doc1"]),
            ],
            non_taxonomic_relations=[
                RelationResult("sensor", "part-of", "system", source_doc_id="doc2"),
                RelationResult("temperature sensor", "measures", "temperature"),
            ],
        )
        terms = [ExtractedTerm("temperature sensor", ["doc1"])]

        result = output.to_challenge_format(terms, ["doc1", "doc2", "doc3"])

        self.assertEqual([doc["id"] for doc in result], ["doc1", "doc2", "doc3"])
        triples_by_doc = {
            doc["id"]: {tuple(triple) for triple in doc["primitive-ontology-triples"]}
            for doc in result
        }
        self.assertEqual(
            triples_by_doc["doc1"],
            {
                ("temperature sensor", "instance-of", "sensor"),
                ("sensor", "is-a", "device"),
                ("temperature sensor", "measures", "temperature"),
            },
        )
        self.assertEqual(triples_by_doc["doc2"], {("sensor", "part-of", "system")})
        self.assertEqual(triples_by_doc["doc3"], set())


class LLMJsonParsingTests(unittest.TestCase):
    def test_extract_json_strips_thinking_and_markdown_fences(self) -> None:
        text = '<think>internal reasoning</think>\n```json\n{"terms": ["sensor"]}\n```'
        self.assertEqual(_extract_json(text), {"terms": ["sensor"]})

    def test_extract_json_ignores_leading_and_trailing_text(self) -> None:
        self.assertEqual(_extract_json('Here is the answer: {"ok": true} done'), {"ok": True})

    def test_extract_json_rejects_missing_json_object(self) -> None:
        with self.assertRaises(LLMError):
            _extract_json("no structured response")


class EvaluationTests(unittest.TestCase):
    def test_submission_eval_normalizes_case_and_underscores(self) -> None:
        predicted = [
            {"id": "doc1", "primitive-ontology-triples": [["Frustum of Cone", "is-a", "Cone"]]},
            {"id": "doc2", "primitive-ontology-triples": []},
        ]
        gold = [
            {"id": "doc1", "primitive-ontology-triples": [["frustum_of_cone", "is-a", "cone"]]},
            {"id": "doc2", "primitive-ontology-triples": []},
        ]

        per_doc = eval_submission_per_doc(predicted, gold)

        self.assertEqual(per_doc["doc1"]["f1"], 1.0)
        self.assertEqual(per_doc["doc2"]["f1"], 1.0)
        self.assertEqual(macro_average(per_doc)["f1"], 1.0)
        self.assertEqual(micro_average(per_doc)["f1"], 1.0)


class LoaderTests(unittest.TestCase):
    def test_run_step_loaders_parse_intermediate_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.jsonl"
            corpus.write_text(
                json.dumps({"id": "doc1", "text": "A sensor measures temperature."}) + "\n",
                encoding="utf-8",
            )
            s1 = root / "s1.json"
            s1.write_text(
                json.dumps({"terms": [{"text": "sensor", "source_doc_ids": ["doc1"]}]}),
                encoding="utf-8",
            )
            s2 = root / "s2.json"
            s2.write_text(
                json.dumps({"types": [{"text": "device", "source_doc_ids": ["doc1"]}]}),
                encoding="utf-8",
            )
            s3 = root / "s3.json"
            s3.write_text(
                json.dumps(
                    {
                        "term_typings": [
                            {"term": "sensor", "types": ["device"], "source_doc_ids": ["doc1"]}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            s4 = root / "s4.json"
            s4.write_text(
                json.dumps(
                    {
                        "taxonomic_relations": [
                            {"parent": "device", "child": "sensor", "source_doc_ids": ["doc1"]}
                        ]
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(_load_corpus(str(corpus))[0]["id"], "doc1")
            self.assertEqual(_load_terms(str(s1))[0].text, "sensor")
            self.assertEqual(_load_types(str(s2))[0].text, "device")
            self.assertEqual(_load_term_typings(str(s3))[0].types, ["device"])
            self.assertEqual(_load_taxonomy(str(s4))[0].parent, "device")


class ContextHelperTests(unittest.TestCase):
    def test_context_helpers_find_title_and_sentence(self) -> None:
        text = "Title: Smart home\nA temperature sensor measures room temperature. Other text."

        self.assertEqual(_doc_title(text), "Smart home")
        self.assertEqual(
            _containing_sentence(text, "temperature sensor"),
            "Title: Smart home A temperature sensor measures room temperature.",
        )
        self.assertEqual(
            _find_context_sentence("temperature sensor", text),
            "Title: Smart home\nA temperature sensor measures room temperature.",
        )

    def test_build_contexts_uses_full_document_or_sentence_mode(self) -> None:
        term = ExtractedTerm("sensor", ["doc1"])
        text = "Title: Demo\nThe sensor observes humidity."

        contexts, doc_context, chars = _build_contexts([term], text, "full")
        self.assertIsNone(contexts)
        self.assertEqual(doc_context, text)
        self.assertEqual(chars, 150)

        contexts, doc_context, chars = _build_contexts([term], text, "title_sentence")
        self.assertEqual(contexts, ["Title: Demo. Title: Demo The sensor observes humidity."])
        self.assertIsNone(doc_context)
        self.assertEqual(chars, 400)


if __name__ == "__main__":
    unittest.main()
