#!/bin/bash
#SBATCH --job-name=s5_finetune
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#
# QLoRA fine-tuning of S5 non-taxonomic relation extraction (Qwen3-14B by default).
#
# Prereq (ONE TIME): create the training env with the GPU deps —
#   conda create -n s5ft python=3.11 -y && conda activate s5ft
#   pip install torch --index-url https://download.pytorch.org/whl/cu121   # match node CUDA
#   pip install -r scripts/requirements-finetune.txt
#
# Launch:           sbatch scripts/finetune_s5.slurm
# Override params:  FT_EPOCHS=2 FT_OUT=models/s5_qlora_e2 sbatch scripts/finetune_s5.slurm
#   FT_MODEL  base model path   (default Qwen3-14B)
#   FT_TRAIN/FT_VAL             (default data/s5_ft/{train,val}.jsonl)
#   FT_OUT    adapter out dir   (default models/s5_qlora)
#   FT_EPOCHS/FT_BATCH/FT_GRADACCUM/FT_MAXLEN
#   FT_LORA_R/FT_LORA_ALPHA     (default 16/32; r32 probe: FT_LORA_R=32 FT_LORA_ALPHA=64)
#   PIPELINE_CONDA_ENV          (default s5ft)

set -euo pipefail
mkdir -p scripts/logs

CONDA_ENV="${PIPELINE_CONDA_ENV:-s5ft}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="${FT_MODEL:-Qwen/Qwen3-14B}"
TRAIN="${FT_TRAIN:-data/s5_ft/train.jsonl}"
VAL="${FT_VAL:-}"                       # optional; unused in training (eval off)
VAL_ARG=(); [[ -n "$VAL" ]] && VAL_ARG=(--val "$VAL")
OUT="${FT_OUT:-models/s5_qlora}"

echo "========================================"
echo "Job ID  : ${SLURM_JOB_ID:-local}   Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Env     : ${CONDA_ENV}   GPU: ${CUDA_VISIBLE_DEVICES:-?}"
echo "Model   : ${MODEL}"
echo "Train   : ${TRAIN}   Val: ${VAL}"
echo "Output  : ${OUT}"
echo "Start   : $(date)"
echo "========================================"

python scripts/finetune_s5.py \
    --model "${MODEL}" \
    --train "${TRAIN}" "${VAL_ARG[@]}" \
    --output-dir "${OUT}" \
    --epochs "${FT_EPOCHS:-3}" \
    --batch-size "${FT_BATCH:-1}" \
    --grad-accum "${FT_GRADACCUM:-16}" \
    --max-seq-len "${FT_MAXLEN:-4096}" \
    --lora-r "${FT_LORA_R:-16}" --lora-alpha "${FT_LORA_ALPHA:-32}"

echo "Done    : $(date)  → adapter in ${OUT}"
