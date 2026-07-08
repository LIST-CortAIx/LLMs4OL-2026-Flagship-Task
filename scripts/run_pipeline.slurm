#!/bin/bash
#SBATCH --job-name=llms4ol_pipeline
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#
# Run the full ontology learning pipeline S1→S5 on the WHOLE data with the
# default parameters from config.yaml, then assemble the challenge submission.
#
# Every step uses its config.yaml defaults (best retained configs) — no per-step
# overrides here. S1 picks up its classifier + corrected/terms index from config;
# S2 runs non-thinking; S3 snaps to the S2 vocab + frequent-type prior; S4 uses
# the configured LLM mode from S2 types; S5 infers domain then extracts relations.
#
# ─── Usage ─────────────────────────────────────────────────────────────────
#
#   Final submission (whole-data test input → submission.json):
#     sbatch scripts/run_pipeline.slurm
#     bash   scripts/run_pipeline.slurm          # run directly (server must be up)
#
#   Point at the vLLM server (or set base_url in config.yaml):
#     LLM_BASE_URL=http://nodeXX:30000/v1 sbatch scripts/run_pipeline.slurm
#
#   Development eval on the split (override paths + enable eval):
#     PIPELINE_INPUT=data/splits/test_corpus.jsonl \
#     PIPELINE_INDEX_DIR=indexes/split_s0_retriever \
#     PIPELINE_GOLD_DIR=data/splits/test_gold \
#     PIPELINE_EVAL=1 \
#     sbatch scripts/run_pipeline.slurm
#
# ─── Env var overrides (all optional) ──────────────────────────────────────
#   PIPELINE_TEST_INPUT   Raw test JSON ([{id,context}])   default data/test_task_a_input.json
#   PIPELINE_INPUT        Corpus JSONL (skips conversion)  default <derived from TEST_INPUT>
#   PIPELINE_OUT_DIR      Output dir                       default outputs
#   PIPELINE_OUTPUT       Final submission path            default <OUT_DIR>/submission.json
#   PIPELINE_INDEX_DIR    Force one index dir for S1–S5    default: per-step config defaults
#   PIPELINE_WHOLE_INDEX  Whole-data index for S4 few-shot default indexes/s0_retriever_full
#   PIPELINE_WORKERS      Parallel LLM calls               default config.yaml (workers)
#   PIPELINE_EVAL         1 = eval each step vs gold       default 0 (no gold for real test)
#   PIPELINE_GOLD_DIR     Gold dir for eval                default data/gold
#   Backend override (for parallel runs on different models — no config.yaml edit):
#   LLM_BASE_URL / LLM_MODEL / LLM_API_KEY   base_url / model / api_key
#   PIPELINE_PROFILE          qwen | gpt-oss | vllm | openai
#   PIPELINE_INSECURE         1 = skip TLS verify (self-signed https, e.g. Minimax)
#   PIPELINE_REASONING_EFFORT low | medium | high   (gpt-oss only)
#   S5 on a SEPARATE endpoint (e.g. the fine-tuned RAG-FT model; S1–S4 stay on the main):
#   S5_BASE_URL   FT vLLM endpoint, e.g. http://nodeXX:PORT/v1  (enables FT mode)
#   S5_MODEL      served model name                 default s5-ft
#   S5_PROFILE    backend profile for that endpoint default qwen (local vLLM)
#                 FT mode also forces --no-thinking --no-domain-inference (FT emits JSON, no domain step)

set -euo pipefail

# ─── Conda + offline ─────────────────────────────────────────────────────────
CONDA_ENV="${PIPELINE_CONDA_ENV:-llms4ol}"
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
[[ -n "${LLM_BASE_URL:-}" ]] && export LLM_BASE_URL
[[ -n "${LLM_MODEL:-}" ]]    && export LLM_MODEL

# Backend overrides forwarded to every LLM step (so parallel runs needn't share
# config.yaml). base_url/model via $LLM_BASE_URL / $LLM_MODEL above; api_key via
# $LLM_API_KEY (read by run_step directly). profile / insecure / reasoning_effort below:
BACKEND_ARGS=()
[[ -n "${PIPELINE_PROFILE:-}" ]]          && BACKEND_ARGS+=(--profile "$PIPELINE_PROFILE")
[[ -n "${PIPELINE_REASONING_EFFORT:-}" ]] && BACKEND_ARGS+=(--reasoning-effort "$PIPELINE_REASONING_EFFORT")
[[ "${PIPELINE_INSECURE:-0}" == "1" ]]    && BACKEND_ARGS+=(--insecure)
[[ -n "${LLM_API_KEY:-}" ]]               && BACKEND_ARGS+=(--api-key "$LLM_API_KEY")

# S5 can run on a separate endpoint (the fine-tuned RAG-FT model) while S1–S4 use the
# main model. If S5_BASE_URL is set, S5 swaps to it with the FT-mode flags (the model
# emits relations JSON directly: no thinking, no domain-inference pre-step). Otherwise
# S5 inherits the same backend as the rest of the pipeline.
S5_ARGS=("${BACKEND_ARGS[@]}")
if [[ -n "${S5_BASE_URL:-}" ]]; then
    # A dedicated S5 endpoint runs gate-free (--relation-classifier "") regardless of
    # config.yaml, so it never inherits the relation-presence gate.
    S5_ARGS=(--base-url "$S5_BASE_URL" --model "${S5_MODEL:-s5-ft}"
             --profile "${S5_PROFILE:-qwen}" --no-thinking --no-domain-inference
             --relation-classifier "")
    [[ -n "${LLM_API_KEY:-}" ]] && S5_ARGS+=(--api-key "$LLM_API_KEY")
fi
# Override the S5 relation-gate threshold per run (else uses config gate_threshold_s5).
# Higher = filter more docs (fewer relations). e.g. aggressive 0.65 vs milder 0.40.
[[ -n "${S5_GATE_THRESHOLD:-}" ]] && S5_ARGS+=(--gate-threshold "$S5_GATE_THRESHOLD")

# ─── Read defaults from config.yaml ──────────────────────────────────────────
_cfg() {
    python3 -c "
import re, sys
key, fallback = sys.argv[1], sys.argv[2]
try:
    cfg = open('config.yaml').read()
    m = re.search(r'^' + re.escape(key) + r':\s*(.+)', cfg, re.MULTILINE)
    print(m.group(1).split('#')[0].strip() if m else fallback)
except Exception:
    print(fallback)
" "$1" "$2"
}

# ─── Scope: whole (final submission) vs split (held-out dev evaluation) ──────
# SCOPE=split runs on the dev split (config.split.yaml, split index/gold/classifiers)
# with evaluation on. SCOPE=whole (default) uses config.yaml and the full-data index.
# Exported so run_step.py picks the matching config file.
SCOPE="${SCOPE:-whole}"; export SCOPE
if [[ "$SCOPE" == "split" ]]; then
    : "${PIPELINE_INPUT:=data/splits/test_corpus.jsonl}"
    : "${PIPELINE_INDEX_DIR:=indexes/split_s0_retriever}"
    : "${PIPELINE_GOLD_DIR:=data/splits/test_gold}"
    : "${PIPELINE_EVAL:=1}"
fi

# ─── Paths / params ──────────────────────────────────────────────────────────
TEST_INPUT="${PIPELINE_TEST_INPUT:-data/test_task_a_input.json}"
OUT_DIR="${PIPELINE_OUT_DIR:-outputs}"
OUTPUT="${PIPELINE_OUTPUT:-$OUT_DIR/submission.json}"
WHOLE_INDEX="${PIPELINE_WHOLE_INDEX:-indexes/s0_retriever_full}"
WORKERS="${PIPELINE_WORKERS:-$(_cfg workers 8)}"
DO_EVAL="${PIPELINE_EVAL:-0}"
GOLD_DIR="${PIPELINE_GOLD_DIR:-data/gold}"

mkdir -p "$OUT_DIR" scripts/logs

# Corpus: use PIPELINE_INPUT if given, else derive a JSONL corpus from TEST_INPUT.
if [[ -n "${PIPELINE_INPUT:-}" ]]; then
    INPUT="$PIPELINE_INPUT"
else
    INPUT="$OUT_DIR/test_corpus.jsonl"
    echo "Converting $TEST_INPUT → $INPUT"
    python3 -c "
import json, sys
data = json.load(open(sys.argv[1]))
with open(sys.argv[2], 'w', encoding='utf-8') as f:
    for d in data:
        f.write(json.dumps({'id': d['id'], 'text': d['context']}, ensure_ascii=False) + '\n')
print(f'  {len(data)} documents')
" "$TEST_INPUT" "$INPUT"
fi

# If a single index dir is forced (dev), pass it to S1–S5; otherwise rely on the
# per-step config defaults (S1 → corrected index, S2/S3/S5 → whole index).
INDEX_ARGS=()
[[ -n "${PIPELINE_INDEX_DIR:-}" ]] && INDEX_ARGS=(--index-dir "$PIPELINE_INDEX_DIR")
S4_INDEX="${PIPELINE_INDEX_DIR:-$WHOLE_INDEX}"

STEP="python scripts/run_step.py"

fmt_time() { local s=$1; printf "%dh %02dm %02ds" $((s/3600)) $(((s%3600)/60)) $((s%60)); }
eval_step() {
    [[ "$DO_EVAL" != "1" ]] && return 0
    local step="$1" pred="$2" gold="$3"
    if [[ -f "$gold" ]]; then $STEP eval --step "$step" --predictions "$pred" --gold "$gold"
    else echo "WARN: gold not found ($gold) — skipping eval for $step"; fi
}

# ─── Header ──────────────────────────────────────────────────────────────────
echo "========================================"
echo "Job ID      : ${SLURM_JOB_ID:-local}"
echo "Node        : ${SLURM_NODELIST:-$(hostname)}"
echo "Input       : $INPUT"
echo "Output      : $OUTPUT"
echo "Index (S1-5): ${PIPELINE_INDEX_DIR:-per-step config defaults}"
echo "Index (S4)  : $S4_INDEX"
echo "Workers     : $WORKERS"
echo "S5 endpoint : ${S5_BASE_URL:+$S5_BASE_URL (model ${S5_MODEL:-s5-ft}, FT mode: no-thinking/no-domain-inf)}${S5_BASE_URL:-same as S1-S4}"
echo "S5 gate thr : ${S5_GATE_THRESHOLD:-config gate_threshold_s5}"
echo "Eval        : $([ "$DO_EVAL" == "1" ] && echo "on (gold=$GOLD_DIR)" || echo "off")"
echo "Start       : $(date)"
echo "========================================"

declare -A STEP_TIMES
T_PIPELINE=$SECONDS

# ─── S4 needs the document index with taxonomic few-shot metadata ────────────
if [[ ! -f "$S4_INDEX/doc_training.json" || ! -f "$S4_INDEX/doc_embeddings.npy" ]]; then
    echo "ERROR: S4 document index missing in $S4_INDEX." >&2
    echo "       Build all indexes first:  python scripts/prepare_data.py --scope whole" >&2
    exit 1
fi
python3 -c "
import json, sys
docs = json.load(open(sys.argv[1], encoding='utf-8'))
if not any(doc.get('taxonomic_pairs') for doc in docs):
    raise SystemExit(
        'ERROR: S4 index has no taxonomic_pairs metadata; rebuild it with '
        'python scripts/prepare_data.py --scope whole'
    )
" "$S4_INDEX/doc_training.json"

# ─── S1 — Term Extraction ────────────────────────────────────────────────────
echo ""; echo "==== S1: Term Extraction — $(date) ===="
T=$SECONDS
$STEP s1 --input "$INPUT" "${INDEX_ARGS[@]}" "${BACKEND_ARGS[@]}" \
    --output "$OUT_DIR/s1_output.json" --workers "$WORKERS"
STEP_TIMES[s1]=$((SECONDS-T)); echo "--- S1: $(fmt_time "${STEP_TIMES[s1]}") ---"
eval_step s1 "$OUT_DIR/s1_output.json" "$GOLD_DIR/s1_gold.json"

# ─── S2 — Type Extraction (non-thinking) ─────────────────────────────────────
echo ""; echo "==== S2: Type Extraction — $(date) ===="
T=$SECONDS
$STEP s2 --input "$INPUT" "${INDEX_ARGS[@]}" "${BACKEND_ARGS[@]}" --no-thinking \
    --output "$OUT_DIR/s2_output.json" --workers "$WORKERS"
STEP_TIMES[s2]=$((SECONDS-T)); echo "--- S2: $(fmt_time "${STEP_TIMES[s2]}") ---"
eval_step s2 "$OUT_DIR/s2_output.json" "$GOLD_DIR/s2_gold.json"

# ─── S3 — Term Typing (snap to S2 vocab) ─────────────────────────────────────
echo ""; echo "==== S3: Term Typing — $(date) ===="
T=$SECONDS
$STEP s3 --s1-input "$OUT_DIR/s1_output.json" --s2-input "$OUT_DIR/s2_output.json" \
    --corpus "$INPUT" "${INDEX_ARGS[@]}" "${BACKEND_ARGS[@]}" \
    --output "$OUT_DIR/s3_output.json" --workers "$WORKERS"
STEP_TIMES[s3]=$((SECONDS-T)); echo "--- S3: $(fmt_time "${STEP_TIMES[s3]}") ---"
eval_step s3 "$OUT_DIR/s3_output.json" "$GOLD_DIR/s3_gold.json"

# ─── S4 — Taxonomy Discovery from S2 types ───────────────────────────────────
echo ""; echo "==== S4: Taxonomy Discovery — $(date) ===="
T=$SECONDS
$STEP s4 --s2-input "$OUT_DIR/s2_output.json" --corpus "$INPUT" \
    --index-dir "$S4_INDEX" --output "$OUT_DIR/s4_output.json" \
    --workers "$WORKERS" --no-thinking "${BACKEND_ARGS[@]}"
STEP_TIMES[s4]=$((SECONDS-T)); echo "--- S4: $(fmt_time "${STEP_TIMES[s4]}") ---"
eval_step s4 "$OUT_DIR/s4_output.json" "$GOLD_DIR/s4_gold.json"

# ─── S5 — Non-Taxonomic Relation Extraction ──────────────────────────────────
echo ""; echo "==== S5: Non-Taxonomic RE — $(date) ===="
T=$SECONDS
$STEP s5 --corpus "$INPUT" \
    --s1-input "$OUT_DIR/s1_output.json" --s2-input "$OUT_DIR/s2_output.json" \
    --input "$OUT_DIR/s3_output.json" "${S5_ARGS[@]}" \
    --output "$OUT_DIR/s5_output.json" --workers "$WORKERS"
STEP_TIMES[s5]=$((SECONDS-T)); echo "--- S5: $(fmt_time "${STEP_TIMES[s5]}") ---"
eval_step s5 "$OUT_DIR/s5_output.json" "$GOLD_DIR/s5_gold.json"

# ─── Assemble submission ─────────────────────────────────────────────────────
echo ""; echo "==== Assembling submission — $(date) ===="
$STEP assemble --corpus "$INPUT" \
    --s1-input "$OUT_DIR/s1_output.json" \
    --s3-input "$OUT_DIR/s3_output.json" \
    --s4-input "$OUT_DIR/s4_output.json" \
    --s5-input "$OUT_DIR/s5_output.json" \
    --output "$OUTPUT"
[[ "$DO_EVAL" == "1" && -f "$GOLD_DIR/submission_gold.json" ]] && \
    $STEP eval --step submission --predictions "$OUTPUT" --gold "$GOLD_DIR/submission_gold.json"

# ─── Timing summary ──────────────────────────────────────────────────────────
T_TOTAL=$((SECONDS-T_PIPELINE))
echo ""; echo "========================================"
echo "Timing summary"
printf "  %-6s  %s\n" "Step" "Duration"
for step in s1 s2 s3 s4 s5; do printf "  %-6s  %s\n" "$step" "$(fmt_time "${STEP_TIMES[$step]}")"; done
printf "  %-6s  %s\n" "TOTAL" "$(fmt_time $T_TOTAL)"
echo ""
echo "Intermediates : $OUT_DIR/s{1,2,3,4,5}_output.json"
echo "Submission    : $OUTPUT"
echo "Done: $(date)"
echo "========================================"
