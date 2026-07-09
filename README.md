# LIST-CortAIx at LLMs4OL 2026 - Option 2 Script Submission

This directory is the flat script-based submission for the LLMs4OL 2026 Flagship Task.
All Python source files are placed directly in this directory, as requested by Option 2.

The primary entry point is `main.py`. It reproduces the same pipeline orchestration as
the original `run_pipeline.slurm`: S1 -> S2 -> S3 -> S4 -> S5 -> assemble.

## Requirements

Use Python 3.11 or 3.12.

```bash
pip install -r requirements.txt
```

The pipeline client calls an already running OpenAI-compatible vLLM server. Start the
LLM server separately, then set the endpoint in `config.yaml` or pass it at runtime.

Model serving/deployment is intentionally outside this flat Option 2 package. For the
Slurm/Apptainer vLLM deployment scripts and operational details, refer to the complete
project repository, which contains the full package layout and `scripts/serve_vllm.slurm`.

## Rebuild Artifacts

The data, retriever indexes, classifiers and outputs are generated artifacts and are
not expected to be inside the source-only submission zip.

Place the challenge files under `data/`:

```text
data/train_task_a.json
data/test_task_a_input.json
```

Build the whole-training retriever index and gold files:

```bash
python prepare_data.py --scope whole
```

Train the optional classifier gates used by the default config:

```bash
python train_term_classifier.py --whole --threshold 0.5 --output models/term_classifier_full_t05.pkl
python train_s5_relation_classifier.py --whole --output models/s5_relation_classifier_full.pkl
```

## Run the Full Pipeline

With the vLLM server running:

```bash
python main.py \
  --test-input data/test_task_a_input.json \
  --output outputs/submission.json \
  --base-url http://nodeXX:30000/v1 \
  --model Qwen/Qwen3.5-27B
```

If `--input` is not provided, `main.py` converts the raw test JSON
`[{id, context}, ...]` into `outputs/test_corpus.jsonl`.

The final file is:

```text
outputs/submission.json
```

For split-scope development evaluation:

```bash
python split_data.py
python prepare_data.py --scope split
python train_term_classifier.py --output models/term_classifier_split.pkl
python train_s5_relation_classifier.py --output models/s5_relation_classifier_split.pkl
python main.py --scope split --base-url http://nodeXX:30000/v1 --model Qwen/Qwen3.5-27B
```

## Unit Tests

Run:

```bash
python unittest.py
```

These tests are deterministic and do not call the LLM server. They cover final-format
serialization, LLM JSON parsing, evaluation metrics, intermediate-file loaders and
context helpers.

## Package as ZIP

Create the ZIP from inside this directory so the files appear at the archive root.
Exclude generated runtime artifacts if you have already run the pipeline locally:

```bash
zip -r ../LIST-CortAIx-LLMs4OL-2026-option2.zip . \
  -x 'data/*' 'indexes/*' 'models/*' 'outputs/*' '__pycache__/*' '*.pyc'
```

## Script Descriptions

`main.py`: Primary entry point. Runs S1, S2, S3, S4, S5 and assembles the final
challenge-format submission.

`run_step.py`: Step-by-step command runner. It exposes `s0`, `s1`, `s2`, `s3`, `s4`,
`s5`, `assemble` and `eval` subcommands.

`prepare_data.py`: Builds per-step gold files and S0 retriever indexes from the
challenge training data.

`split_data.py`: Creates the reproducible 80/20 development split and split gold files.

`train_term_classifier.py`: Trains the S1 term-presence classifier used to skip
documents predicted to contain no terms.

`train_s5_relation_classifier.py`: Trains the S5 relation-presence classifier used to
skip documents predicted to contain no non-taxonomic relation.

`llm.py`: HTTP client for OpenAI-compatible inference servers, including JSON extraction
from model responses.

`models.py`: Dataclasses for S1-S5 outputs and conversion to the final challenge format.

`prompts.py`: Prompt builders for term extraction, type extraction, term typing,
taxonomy discovery and non-taxonomic relation extraction.

`progress.py`: Small progress-bar helper used by long-running steps.

`pipeline.py`: Programmatic pipeline wrapper around the S1-S5 step implementations.

`evaluate.py`: Per-step evaluation utilities and optional graph-similarity reporting.

`learner.py`: Optional OntoLearner wrapper around the pipeline.

`s0_retriever.py`: Builds and loads embedding retrievers for few-shot examples and
term-type pair retrieval.

`s1_term_extraction.py`: Step 1, per-document term extraction.

`s2_type_extraction.py`: Step 2, per-document type extraction.

`s3_term_typing.py`: Step 3, RAG-based assignment of semantic types to extracted terms.

`s4_taxonomy.py`: Step 4, discovery of taxonomic `is-a` relations.

`s5_relations.py`: Step 5, extraction of non-taxonomic relations between types.

`unittest.py`: Unit tests for deterministic core components.

`config.yaml`: Whole-scope default configuration used for final submission runs.

`config.split.yaml`: Split-scope configuration used for development evaluation.

`requirements.txt`: Runtime Python dependencies.
