#!/bin/bash
#SBATCH --job-name=s5_ft_serve
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#
# PERSISTENT serve of the fine-tuned S5 model (Qwen3-14B base + LoRA adapter) via vLLM,
# for Run 2 of the submission (RAG-FT, gate-free). Unlike serve_eval_ft_s5.slurm this
# binds 0.0.0.0 and STAYS ALIVE (no eval, no teardown) so the pipeline — running on a
# DIFFERENT node — can reach it via S5_BASE_URL.
#
# Submit:
#   FT_ADAPTER=models/s5_qlora_whole sbatch scripts/serve_ft_s5.slurm
#   # r32 adapter: add FT_LORA_RANK=32
#
# Then read "S5_BASE_URL" from the job log and launch the pipeline (S1-S4 on the main
# model, S5 on this FT endpoint — FT mode auto-disables the gate):
#   LLM_BASE_URL=http://<main-node>:30000/v1 \
#   S5_BASE_URL=http://<this-node>:30001/v1 \
#   sbatch scripts/run_pipeline.slurm
#
# Env overrides:
#   FT_BASE      base model           default Qwen3-14B
#   FT_ADAPTER   LoRA adapter dir     default models/s5_qlora_whole
#   FT_LORA_RANK max LoRA rank        default 16 (set 32 for an r32 adapter)
#   VLLM_SIF     apptainer image      default the project's vllm-openai.sif
#   VLLM_PORT    server port          default 30001 (keep != the main model's 30000)
#   VLLM_MAXLEN  max model len        default 16384 (RAG few-shot prompts are long)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p scripts/logs

BASE="${FT_BASE:-Qwen/Qwen3-14B}"
ADAPTER="$(readlink -f "${FT_ADAPTER:-models/s5_qlora_whole}")"
SIF="${VLLM_SIF:-vllm-openai.sif}"
PORT="${VLLM_PORT:-30001}"
MAXLEN="${VLLM_MAXLEN:-16384}"
LORA_RANK="${FT_LORA_RANK:-16}"

echo "========================================"
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : ${SLURM_NODELIST:-$(hostname)}"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Base       : $BASE"
echo "Adapter    : $ADAPTER  (max-lora-rank $LORA_RANK)"
echo "Port       : $PORT     MaxLen: $MAXLEN"
echo "Start      : $(date)"
echo "S5_BASE_URL: http://${SLURM_NODELIST:-$(hostname)}:${PORT}/v1   (model: s5-ft)"
echo "========================================"
[[ -d "$ADAPTER" ]] || { echo "ERROR: adapter dir not found: $ADAPTER" >&2; exit 1; }

# Serve base + LoRA, bound to 0.0.0.0 so other nodes can connect. No reasoning-parser:
# the FT model emits the relations JSON directly (trained with enable_thinking=False).
apptainer run --nv \
    --bind "$BASE:$BASE" --bind "$ADAPTER:$ADAPTER" --bind /tmp:/tmp \
    --env "HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_DATASETS_OFFLINE=1" \
    "$SIF" \
    --model "$BASE" --served-model-name base \
    --enable-lora --lora-modules "s5-ft=$ADAPTER" --max-lora-rank "$LORA_RANK" \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len "$MAXLEN" --dtype bfloat16 --trust-remote-code
