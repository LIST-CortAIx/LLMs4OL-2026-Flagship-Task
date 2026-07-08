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
# Submit:
#   sbatch scripts/serve_vllm.slurm
#
# Once running, submit the pipeline:
#   sbatch scripts/run_pipeline.slurm

set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-27B}"
PORT="${VLLM_PORT:-30000}"
SIF="${VLLM_SIF:-vllm-openai.sif}"

echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Model    : $MODEL"
echo "Port     : $PORT"
echo "SIF      : $SIF"
echo "Start    : $(date)"
echo "Base URL : http://${SLURM_NODELIST}:${PORT}/v1"
echo "========================================"

apptainer run --nv \
    --bind "$MODEL:$MODEL" \
    --bind /tmp:/tmp \
    --env "HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_DATASETS_OFFLINE=1" \
    "$SIF" \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --language-model-only \
    --enable-prefix-caching \
    --dtype bfloat16 \
    --trust-remote-code
