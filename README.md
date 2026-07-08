# LIST-CortAIx at LLMs4OL 2026 — Flagship Task

This repository is the **LIST-CortAIx** submission to the **Flagship Task** of the
**3rd LLMs4OL Challenge @ ISWC 2026** (Bari, Italy, 25–29 October 2026). It builds an
ontology end-to-end from raw text through five stages — term extraction, type extraction,
term typing, taxonomy discovery, and non-taxonomic relation extraction — driven by a single
mid-sized LLM served via vLLM.

```
S0    Build retriever indices (doc few-shot index + S3 term-type pair index)
S1    Per-document term extraction with RAG few-shot from training data
S2    Per-document type extraction with RAG few-shot from training data (independent of S1)
S3    Term Typing: RAG-based kNN over gold (term, type) pairs
S4    Taxonomy Discovery: is-a edges by LLM candidate verification
S5    Non-Taxonomic RE: remaining semantic relations between types
```

---

## Architecture

The pipeline runs in two separate components:

```
[ vLLM server (Apptainer) ]  ←HTTP→  [ Pipeline client ]
  GPU node (Slurm)                      any CPU node
```

- **vLLM server**: serves the LLM via Apptainer container on a GPU node, OpenAI-compatible HTTP API.
- **Pipeline client**: makes HTTP calls to the server — no GPU, no CUDA, no torch required.

### Prerequisites

| Component | Needs |
|---|---|
| Pipeline client | Python 3.11–3.12 (conda recommended); the packages in `requirements.txt`. No GPU. |
| LLM server | A GPU large enough for the chosen model, and **one of**: [Apptainer](https://apptainer.org/) (system-level runtime — not a pip package; e.g. `module load apptainer`) to run the provided SLURM script, **or** `pip install vllm` to run vLLM directly (see [Serving the model](#serving-the-model)). Slurm is optional. |

---

## Installation (pipeline client only)

```bash
conda create -n llms4ol python=3.12 -y
conda activate llms4ol
pip install -e .
pip install -r requirements.txt
```

---

## Serving the model

Every LLM step (S1–S5, and the full pipeline) talks to a vLLM server over HTTP, so start it
**first**. Any OpenAI-compatible vLLM endpoint works; the client steps themselves need no GPU.
(S0 and data preparation are CPU-only and do not need the server.)

The provided SLURM script runs vLLM through an Apptainer container. Build the image once from
the official public vLLM image (no image is shipped in this repo):

```bash
apptainer build vllm-openai.sif docker://vllm/vllm-openai:latest
# the scripts look for ./vllm-openai.sif, or set $VLLM_SIF to another path
```

Then launch the server:

```bash
sbatch scripts/serve_vllm.slurm
tail -f scripts/logs/llms4ol_serve_<jobid>.out
# Look for: Base URL : http://nodeXX:30000/v1
```

The model defaults to `Qwen/Qwen3.5-27B` (override with `$VLLM_MODEL`). When it is a Hugging
Face repo id, the first run **downloads the weights** from the Hub and caches them under
`$HF_HOME` (default `./.hf_cache`); when it is a **local directory** (already downloaded), that
directory is bound into the container and served offline — no download:

```bash
# serve a locally-available model, still addressed by a clean name from the client
VLLM_MODEL=/path/to/Qwen3.5-27B VLLM_SERVED_NAME=Qwen/Qwen3.5-27B sbatch scripts/serve_vllm.slurm
```

`$VLLM_SERVED_NAME` (default: the model value) is the client-facing name; keep `config.yaml`'s
`model` equal to it so a local path doesn't leak into the client config.

No Apptainer / SLURM? Run vLLM directly (`pip install vllm`) with the same flags:

```bash
vllm serve Qwen/Qwen3.5-27B --served-model-name Qwen/Qwen3.5-27B \
  --host 0.0.0.0 --port 30000 --max-model-len 262144 \
  --reasoning-parser qwen3 --enable-prefix-caching --dtype bfloat16 --trust-remote-code
```

> **The `model` value in `config.yaml` must match the server's `--served-model-name`.** The
> client addresses the model by that exact name; a mismatch returns a 404. The default on both
> sides is `Qwen/Qwen3.5-27B`.

Put the resulting URL in `config.yaml` (`base_url`), or pass it per-run via `--base-url` / `$LLM_BASE_URL`.

---

## Configuration

Edit `config.yaml` at the project root to point at the running vLLM server and set sampling parameters:

```yaml
base_url: http://nodeXX:30000/v1
model: Qwen/Qwen3.5-27B
temperature: 0.0          # best across S1/S3/S5 (deterministic); S2 also 0.0
top_p: 0.95
top_k: 20
min_p: 0.0
presence_penalty: 1.5
repetition_penalty: 1.0
max_tokens: 16384         # thinking_budget caps thinking; rest goes to answer
enable_thinking: true
thinking_budget: 4096     # vLLM forces </think> after this many thinking tokens
```

All `run_step.py` commands pick this up automatically. CLI flags and `$LLM_BASE_URL` / `$LLM_MODEL` env vars override `base_url` and `model`. Sampling parameters can be overridden per-run with `--temperature`, `--thinking-budget`, `--no-thinking`, etc.

### Whole vs. split (`--scope`)

The pipeline runs in two scopes:

- `--scope whole` (default) — final-submission config (`config.yaml`): few-shot and indices from all training documents; evaluation off.
- `--scope split` — held-out development config (`config.split.yaml`): few-shot and indices from the split-train documents; evaluated against the split-test gold.

Every `run_step.py` command accepts `--scope`, which selects the matching config file and thus the data, index and classifier paths automatically. `run_pipeline.slurm` takes the same choice via the `SCOPE` environment variable (e.g. `SCOPE=split sbatch scripts/run_pipeline.slurm`). All derived artifacts — `indexes/`, `models/` (classifiers), and the S5 adapter — are **regenerable and not shipped**; build them once with the commands below.

---

## Data preparation

The challenge corpora are provided by the LLMs4OL 2026 organizers and are **not redistributed in this repository** — obtain them from the challenge and place `data/train_task_a.json` (and `data/test_task_a_input.json` for the final test run) under `data/`. Then `prepare_data.py` builds
**the gold files and the retriever indexes** (doc text/terms/types + S3 pair index)
for the chosen scope. Indexes are always built from the train
portion only (no test leakage). Run with the env that has sentence-transformers (`llms4ol`).

```bash
# FINAL SUBMISSION — whole training data: golds + indexes
python scripts/prepare_data.py --scope whole
#   → data/gold/  indexes/s0_retriever_full

# DEVELOPMENT — first make the 80/20 split, then build split golds + indexes
python scripts/split_data.py
python scripts/prepare_data.py --scope split
#   → data/splits/{train,test}_gold/  indexes/split_s0_retriever

# Options:  --scope both  ·  --no-index (gold only)
```

S1 and S5 each use a TF-IDF + logistic-regression **classifier** that skips the LLM on
documents predicted to have no output (term-free for S1, relation-free for S5). Train them
once; the whole-data variants tune their threshold by cross-validation and live in `models/`:

```bash
# S1 term-presence classifier
python scripts/train_term_classifier.py --whole --threshold 0.5 --output models/term_classifier_full_t05.pkl  # whole (submission)
python scripts/train_term_classifier.py --output models/term_classifier_split.pkl                             # split (dev)

# S5 relation-presence classifier
python scripts/train_s5_relation_classifier.py --whole --output models/s5_relation_classifier_full.pkl        # whole (submission)
python scripts/train_s5_relation_classifier.py --output models/s5_relation_classifier_split.pkl               # split (dev)
```

`prepare_data.py --scope whole` produces:

| Path | Content |
|---|---|
| `data/corpus.jsonl` | One `{"id", "text"}` per line — full corpus |
| `data/gold/` | Per-step gold (s1–s5 + submission) |
| `indexes/s0_retriever_full/` | Doc index (text/terms/types) + S3 term-type pair index |
| `models/term_classifier_full_t05.pkl` | S1 term-presence classifier (whole-data, threshold 0.5) |

`split_data.py` produces (in `data/splits/`, gitignored):

| File | Content |
|---|---|
| `data/splits/train.json` | 80% of documents — training set (3,444 docs) |
| `data/splits/test.json` | 20% of documents — evaluation set (859 docs) |
| `data/splits/train_corpus.jsonl` | Train corpus for S0 index building |
| `data/splits/test_corpus.jsonl` | Test corpus for pipeline inference |
| `data/splits/test_gold/` | Per-step gold files for the test set |

The split is stratified (preserves instance-of ratio) with fixed seed=42 for reproducibility.

---

## Running the pipeline

With the model server running (see [Serving the model](#serving-the-model)) and the indexes
built, run the whole pipeline end to end:

```bash
# Final submission (whole training data) — config.yaml defaults
sbatch scripts/run_pipeline.slurm

# Development run on the held-out split — config.split.yaml, evaluates each step vs gold
SCOPE=split sbatch scripts/run_pipeline.slurm
```

`SCOPE` selects the scope for the whole pipeline just as `--scope` does per step: `whole`
(default) uses `config.yaml`, the whole-data index and the challenge test input; `split`
uses `config.split.yaml`, runs on the split test corpus and turns on per-step evaluation
against gold. In both cases it runs **S1→S5 then assemble** with each step's config defaults.
It converts the raw test input (`data/test_task_a_input.json`, a `[{id, context}]` list) to a
JSONL corpus and writes the submission. S1 uses its classifier + the retriever index; S2 runs
non-thinking; S3 snaps to the S2 vocab; S4 verifies is-a candidates with the LLM; S5 infers the
domain then extracts relations. A per-step timing summary is printed at the end.

Intermediate outputs are saved to `outputs/s{1-5}_output.json`. Final challenge submission is written to `outputs/submission.json`:

```json
[{"id": "...", "primitive-ontology-triples": [["child", "is-a", "parent"], ...]}, ...]
```

#### Environment overrides

`SCOPE=split` already sets the split input, index, gold and per-step evaluation. The
variables below give finer control on top of that (custom corpus, output dir, workers, etc.):

| Variable | Default | Notes |
|---|---|---|
| `PIPELINE_TEST_INPUT` | `data/test_task_a_input.json` | Raw test JSON (`[{id, context}]`); converted to a corpus |
| `PIPELINE_INPUT` | derived from `TEST_INPUT` | Corpus JSONL (set this to skip conversion) |
| `PIPELINE_OUT_DIR` | `outputs` | Output directory |
| `PIPELINE_OUTPUT` | `<OUT_DIR>/submission.json` | Final submission path |
| `PIPELINE_INDEX_DIR` | per-step config defaults | Force one index dir for S1–S5 (dev) |
| `PIPELINE_WHOLE_INDEX` | `indexes/s0_retriever_full` | Default S0 index for S4 (when `PIPELINE_INDEX_DIR` unset) |
| `PIPELINE_WORKERS` | from `config.yaml` | Parallel LLM calls per step |
| `PIPELINE_EVAL` | `0` | `1` = evaluate each step vs gold (needs gold) |
| `PIPELINE_GOLD_DIR` | `data/gold` | Gold dir used when `PIPELINE_EVAL=1` |
| `PIPELINE_CONDA_ENV` | `llms4ol` | Conda environment to activate |
| `LLM_BASE_URL` | from `config.yaml` | LLM server URL |
| `LLM_MODEL` | from `config.yaml` | Model name/path |
| `LLM_API_KEY` | from `config.yaml` | Endpoint API key |
| `PIPELINE_PROFILE` | from `config.yaml` | Backend profile: `qwen` \| `gpt-oss` \| `vllm` \| `openai` |
| `PIPELINE_INSECURE` | `0` | `1` = skip TLS verify (self-signed https, e.g. Minimax) |
| `PIPELINE_REASONING_EFFORT` | — | `low` \| `medium` \| `high` (gpt-oss only) |

All sampling/k parameters come from `config.yaml`; override them there or per-step with `run_step.py` flags. The backend (URL/model/key/profile/TLS/reasoning) is also fully env-overridable via the variables in the table above.

---

## Step-by-step testing

Once `config.yaml` is set, test each step interactively from any node:

```bash
mkdir -p outputs
```

### S0 — Build Retriever Indices

Run once before S1, S2, S3, and S4 (CPU-only, no LLM needed). Builds the retriever
indices in the same directory when `--training-data` is provided:

- **Pair index** (`embeddings.npy` + `pairs.json`) — cosine-similarity kNN over gold
  (term, type) pairs, used by S3 for RAG-based typing.
- **Doc index** (`doc_training.json` + three embedding views of each training document) —
  used by S1/S2/S4/S5 for RAG few-shot retrieval. The three views back the `retriever_mode`
  option: `doc_embeddings.npy` (full **text**), `doc_embeddings_terms.npy` (its **terms**),
  and `doc_type_embeddings.npy` (its **types**).

```bash
# Train/test split (development evaluation — TRAIN docs only, no leakage)
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python scripts/run_step.py s0 \
    --gold data/splits/train_gold/s3_gold.json \
    --index-dir indexes/split_s0_retriever \
    --training-data data/splits/train.json

# Whole training data (FINAL SUBMISSION — examples drawn from all 4303 docs)
# First build the whole-data gold once:  python scripts/prepare_data.py  (writes data/gold/)
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python scripts/run_step.py s0 \
    --gold data/gold/s3_gold.json \
    --index-dir indexes/s0_retriever_full \
    --training-data data/train_task_a.json
```

> The pair index for S3 **must** be built from the **train** gold (`train_gold` for the
> split, `data/gold` for submission) — never from `test_gold` (that leaks the answers).

Each step below shows its command and its main parameters; the values are the best
configuration and live in `config.yaml` (override there or with the per-step flag).

### S1 — Term Extraction

Per-document term extraction (one LLM call per doc). A term-presence classifier drops
term-free documents so the LLM only runs on likely term-bearing ones.

Parameters (`config.yaml`):
- `k_examples_s1` — term-bearing few-shot examples per doc; `10` (best)
- `k_empty_s1` — term-free examples (for `similarity_two_pools`); `10` (best)
- `example_selection_s1` — `similarity_two_pools` (best) | `similarity_all_docs`
- `retriever_mode_s1` — `text` (best) | `terms`
- `cot_s1` — `false` (best) | `true` (chain-of-thought prompt)
- `classifier_s1` — term-presence classifier path (empty = disabled)

```bash
python scripts/run_step.py s1 --scope split \
    --input data/splits/test_corpus.jsonl \
    --output outputs/s1_output.json
python scripts/run_step.py eval --step s1 --predictions outputs/s1_output.json --gold data/splits/test_gold/s1_gold.json
```

### S2 — Type Extraction

Per-document type extraction (independent of S1), run **non-thinking** (`--no-thinking`).

Parameters (`config.yaml`):
- `k_examples_s2` — retrieved example documents; `20` (best)
- `s2_example_selection` — `similarity` (best) | `random` | `representative` | `representative_diverse` | `type_similarity` | `doc_then_type_similarity` | `hybrid_similarity`
- `retriever_mode_s2` — `text` (best) | `terms`
- `cot_s2` — `true` (best) | `false`

```bash
python scripts/run_step.py s2 --scope split \
    --input data/splits/test_corpus.jsonl \
    --no-thinking \
    --output outputs/s2_output.json
python scripts/run_step.py eval --step s2 --predictions outputs/s2_output.json --gold data/splits/test_gold/s2_gold.json
```

### S3 — Term Typing

Free generation grounded on the full document, with RAG over gold (term, type) pairs and
vocabulary snapping to the S2 types.

Parameters (`config.yaml`):
- `k_s3` — nearest gold (term, type) pairs as few-shot; `10` (best)
- `context_s3` — `full` (best) | `sentence` | `title_sentence` | `none`
- `batch_size_s3` — `0` (best, one LLM call per document)
- `snap_s3` — `true` (best) | `false` (snap out-of-vocab types to the nearest S2 type)
- `augment_frequent_s3` — frequent-type fallback pool size; `10` (best), 0 = off

```bash
python scripts/run_step.py s3 --scope split \
    --s1-input outputs/s1_output.json \
    --s2-input outputs/s2_output.json \
    --corpus data/splits/test_corpus.jsonl \
    --output outputs/s3_output.json
python scripts/run_step.py eval --step s3 --predictions outputs/s3_output.json --gold data/splits/test_gold/s3_gold.json
```

### S4 — Taxonomy Discovery

Candidate verification: for each document, form candidate (child, parent) pairs from the S2
types and ask the LLM to confirm the valid is-a links, guided by contrastive few-shot examples.

Parameters (`config.yaml`):
- `s4_mode` — LLM verification mode; `fewshot_grouped_pos_neg_child_batches` (best)
- `k_examples_s4` — few-shot documents per call; `3` (best)
- `s4_max_positive_examples` / `s4_max_negative_examples` — examples per call; `20` / `20` (best)
- `batch_size_s4` — candidate pairs per LLM call; `10` (best)
- `s4_example_seed` — example-selection seed; `13` (best)

```bash
python scripts/run_step.py s4 --scope split \
    --s2-input outputs/s2_output.json \
    --corpus data/splits/test_corpus.jsonl \
    --output outputs/s4_output.json
python scripts/run_step.py eval --step s4 --predictions outputs/s4_output.json --gold data/splits/test_gold/s4_gold.json
```

### S5 — Non-Taxonomic Relation Extraction

Type-constrained relation extraction with full-pool few-shot, an inferred domain description,
and a relation-presence classifier gate.

Parameters (`config.yaml`):
- `k_examples_s5` — nearest training documents as few-shot; `10` (best), 0 = off
- `prompt_v2_s5` — `true` (best) | `false`
- `full_pool_s5` — `true` (best) | `false` (draw few-shot from all training docs)
- `domain_inference_s5` — `true` (best) | `false`
- `retriever_mode_s5` — `text` (best) | `types`
- `relation_classifier_s5` — gate classifier path (empty = disabled)
- `gate_threshold_s5` — keep documents scored ≥ threshold; `0.65` (best)

```bash
python scripts/run_step.py s5 --scope split \
    --corpus data/splits/test_corpus.jsonl \
    --s1-input outputs/s1_output.json \
    --s2-input outputs/s2_output.json \
    --input outputs/s3_output.json \
    --output outputs/s5_output.json
python scripts/run_step.py eval --step s5 --predictions outputs/s5_output.json --gold data/splits/test_gold/s5_gold.json
```

### Assemble submission

Build the challenge-format submission from intermediate step outputs without re-running any LLM inference:

```bash
python scripts/run_step.py assemble \
    --corpus data/splits/test_corpus.jsonl \
    --s1-input outputs/s1_output.json \
    --s3-input outputs/s3_output.json \
    --s4-input outputs/s4_output.json \
    --s5-input outputs/s5_output.json \
    --output outputs/submission.json

# Evaluate submission against gold (includes Graph Similarity Score)
python scripts/run_step.py eval --step submission \
    --predictions outputs/submission.json \
    --gold data/splits/test_gold/submission_gold.json
```

The submission eval prints both the per-document P/R/F1 summary and the **Graph Similarity Score** — the organizer metric combining:
- **Edge F1**: exact triple match (precision, recall, F1)
- **Neighborhood Similarity**: Jaccard over per-node outgoing (predicate, object) sets
- **Taxonomy Similarity**: Jaccard over ancestor sets for is-a edges only
- **Final score** = (Edge F1 + Neighborhood Similarity + Taxonomy Similarity) / 3

Prints document count, total triples, and non-empty document count.

## Final submission (whole training data)

`scripts/run_pipeline.slurm` automates everything below (S1→S5 + assemble on the
whole-data test input with config defaults). The manual steps are equivalent.

For the official test set, few-shot examples are drawn from **all 4303 training docs**
(index `indexes/s0_retriever_full`, built in S0 above). No gold exists for the real
test set, so there is no eval step — only run + assemble.

```bash
# 0. One-time: golds + indexes for the whole training data
python scripts/prepare_data.py --scope whole

# 1. Convert the challenge test input ({id,context} list) to a JSONL corpus
python -c "import json; [print(json.dumps({'id':d['id'],'text':d['context']})) \
    for d in json.load(open('data/test_task_a_input.json'))]" > data/test_task_a_corpus.jsonl

# 2. Run each step on the test corpus, examples from the whole-data index
#    (index_dir defaults to indexes/s0_retriever_full in config.yaml)
python scripts/run_step.py s1 --input data/test_task_a_corpus.jsonl --output outputs/sub_s1.json
python scripts/run_step.py s2 --input data/test_task_a_corpus.jsonl --no-thinking --output outputs/sub_s2.json
python scripts/run_step.py s3 --s1-input outputs/sub_s1.json --s2-input outputs/sub_s2.json \
    --corpus data/test_task_a_corpus.jsonl --output outputs/sub_s3.json
# S4 (taxonomy) — LLM candidate verification
python scripts/run_step.py s4 --s2-input outputs/sub_s2.json \
    --corpus data/test_task_a_corpus.jsonl --output outputs/sub_s4.json
python scripts/run_step.py s5 --corpus data/test_task_a_corpus.jsonl \
    --s1-input outputs/sub_s1.json --s2-input outputs/sub_s2.json \
    --input outputs/sub_s3.json --output outputs/sub_s5.json

# 3. Assemble the submission
python scripts/run_step.py assemble --corpus data/test_task_a_corpus.jsonl \
    --s1-input outputs/sub_s1.json --s3-input outputs/sub_s3.json \
    --s4-input outputs/sub_s4.json --s5-input outputs/sub_s5.json \
    --output outputs/submission.json
```

All step parameters default to the best retained configs in `config.yaml`; override any
with the documented CLI flags (e.g. `--k`, `--context`, `--no-thinking`).

---

## Common flags

| Flag | Default | Notes |
|---|---|---|
| `--base-url` | from `config.yaml` | vLLM server URL |
| `--model` | from `config.yaml` | Model path as registered in the server |
| `--workers` | `1` (slurm: `4`) | Parallel HTTP calls to the server |
| `--max-tokens` | `16384` | Max tokens per LLM response (thinking + answer combined) |
| `--thinking-budget` | from `config.yaml` | vLLM forces `</think>` after this many thinking tokens |
| `--no-thinking` | off | Disable Qwen3.5 thinking mode (faster, lower quality) |
| `--batch-size` | varies | Items per LLM call (S3: terms / S4: type pairs) |
| `--k-examples` | from `config.yaml` (S1: 15, S2: 20) | Training docs retrieved per document for RAG few-shot |
| `--retriever-mode` | from `config.yaml` (S1: `text`) | `text` (full doc), `terms`, or `types` (keyed) retrieval |
| `--classifier` | from `config.yaml` | S1: term/type-only classifier pre-filter (`.pkl`) |
| `--no-snap` | off | S3: disable snapping OOV predicted types to the nearest S2 type |
| `--augment-frequent` | from `config.yaml` (S3: 10) | S3: frequent-type prior block size (0 = off) |
| `--index-dir` | from `config.yaml` | S0 index dir (S4/S5 few-shot retrieval) |
| `--token-overlap` | `0.0` | S4: min token overlap F1 between child/parent labels (0.0 = disabled) |
| `--s4-mode` | from `config.yaml` | S4 LLM mode; default is `fewshot_grouped_pos_neg_child_batches` |
| `--cluster-types` | off | S4 (LLM approach): semantic clustering pre-step |
| `--n-clusters` | `10` | S4 (LLM approach): target cluster count |
| `--no-depth-pass` | — | S4 (LLM approach): skip depth re-prompt pass |
| `--no-domain-inference` | — | S5: disable domain inference pre-step (P1) |
| `--temperature` | from `config.yaml` | Sampling temperature |
| `--top-p` | from `config.yaml` | Nucleus sampling top-p |
| `--top-k` | from `config.yaml` | Top-k sampling |
| `--min-p` | from `config.yaml` | Min-p sampling |
| `--presence-penalty` | from `config.yaml` | Presence penalty |
| `--repetition-penalty` | from `config.yaml` | Repetition penalty |

---

## Fine-tuning S5 (optional)

S5 has an optional fine-tuned variant — a QLoRA adapter on Qwen3-14B. It is regenerable
and **not shipped**. Fine-tuning needs a GPU node and a separate environment:

```bash
pip install -r scripts/requirements-finetune.txt
```

Build the chat-format training data (uses the gold and S0 index for the scope), then train
— split for development, whole for the submission:

```bash
# 1. Build SFT data
python scripts/prepare_s5_finetune.py --scope split --few-shot --oversample-pos 5 --out-dir data/s5_ft_split       # dev
python scripts/prepare_s5_finetune.py --scope whole --few-shot --out-dir data/s5_ft_whole                    # submission

# 2. Train the QLoRA adapter (Qwen3-14B base, 4-bit, r=16)
python scripts/finetune_s5.py --train data/s5_ft_split/train.jsonl       --output-dir models/s5_qlora_split         # dev
python scripts/finetune_s5.py --train data/s5_ft_whole/train.jsonl --output-dir models/s5_qlora_whole   # submission
```

Serve the adapter with vLLM (it is exposed as model `s5-ft`) and point S5 at that endpoint:

```bash
FT_ADAPTER=models/s5_qlora_whole sbatch scripts/serve_ft_s5.slurm
# then run S5 against the FT endpoint (S5 stages only; S1-S4 stay on the main model):
S5_BASE_URL=http://<ft-node>:30002/v1 S5_MODEL=s5-ft bash scripts/run_pipeline.slurm
```

---

## Project layout

```
config.yaml             Whole-scope config (paths + sampling; best values)
config.split.yaml       Split-scope config (--scope split / SCOPE=split)
requirements.txt        Pipeline-client dependencies
ontology_learning/      Importable pipeline package
  llm.py                LLMClient (HTTP wrapper around the vLLM server)
  models.py             Dataclasses for each step + challenge serialization
  prompts.py            Prompt builders for S1–S5
  evaluate.py           Per-step and graph-similarity evaluation
  learner.py            Few-shot example bundling
  pipeline.py           End-to-end pipeline entry
  steps/                s0_retriever + s1..s5 step implementations
data/
  train_task_a.json       Raw challenge training set (tracked)
  test_task_a_input.json  Raw challenge test input (tracked)
indexes/                Committed empty; S0 retriever indexes are built here
models/                 Committed empty; classifiers + S5 adapter are trained here
scripts/
  prepare_data.py                  Build gold files + S0 retriever indexes (per scope)
  split_data.py                    Reproducible 80/20 train/test split (seed=42)
  train_term_classifier.py         S1 term-presence classifier
  train_s5_relation_classifier.py  S5 relation-presence classifier
  run_step.py                      Step-by-step runner (--scope) + eval + assemble
  run_pipeline.slurm               Full pipeline (Slurm; SCOPE=split|whole)
  serve_vllm.slurm                 Start the vLLM server (Apptainer, GPU node)
  prepare_s5_finetune.py           Build S5 SFT data
  finetune_s5.py / finetune_s5.slurm   S5 QLoRA fine-tuning
  serve_ft_s5.slurm                Serve the S5 adapter (vLLM LoRA)
  requirements-finetune.txt        Fine-tuning dependencies

# Generated, gitignored contents (regenerable; the indexes/ and models/ dirs are
#   committed empty): data/corpus.jsonl, data/gold/, data/splits/, indexes/*,
#   models/* (classifiers + S5 adapter), outputs/ (step outputs + submission).
```

---

## License

Released under the Apache License 2.0 — see [LICENSE](LICENSE).
