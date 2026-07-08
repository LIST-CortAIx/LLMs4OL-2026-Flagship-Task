#!/bin/bash
#SBATCH --job-name=llms4ol_serve
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
# Start a vLLM server (OpenAI-compatible API) via Apptainer.
# The server stays alive until the job time limit or manual cancellation.
#
# Build the image once from the official public vLLM image (not shipped):
#   apptainer build vllm-openai.sif docker://vllm/vllm-openai:latest
# Override its path with $VLLM_SIF if it lives elsewhere.
#
# Submit:
#   sbatch scripts/serve_vllm.slurm
#
# Once running, submit the pipeline:
#   sbatch scripts/run_pipeline.slurm

set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-27B}"          # HF repo id or a local model directory
SERVED_NAME="${VLLM_SERVED_NAME:-$MODEL}"        # client-facing name; must match config.yaml's model
PORT="${VLLM_PORT:-30000}"
SIF="${VLLM_SIF:-vllm-openai.sif}"

# A local model directory is bound into the container and served fully offline.
# A Hugging Face repo id (e.g. Qwen/Qwen3.5-27B) is downloaded on first run and
# cached under $HF_HOME (bound so it persists across jobs).
BINDS=(--bind /tmp:/tmp)
ENVS="HF_HUB_OFFLINE=0,TRANSFORMERS_OFFLINE=0"
if [ -d "$MODEL" ]; then
    BINDS+=(--bind "$MODEL:$MODEL")
    ENVS="HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_DATASETS_OFFLINE=1"
else
    HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
    mkdir -p "$HF_HOME"
    BINDS+=(--bind "$HF_HOME:$HF_HOME")
    ENVS="$ENVS,HF_HOME=$HF_HOME"
fi

echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Model    : $MODEL"
echo "Served as: $SERVED_NAME"
echo "Port     : $PORT"
echo "SIF      : $SIF"
echo "Start    : $(date)"
echo "Base URL : http://${SLURM_NODELIST}:${PORT}/v1"
echo "========================================"

apptainer run --nv \
    "${BINDS[@]}" \
    --env "$ENVS" \
    "$SIF" \
    --model "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --language-model-only \
    --enable-prefix-caching \
    --dtype bfloat16 \
    --trust-remote-code
