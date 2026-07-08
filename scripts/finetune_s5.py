"""QLoRA fine-tuning for S5 non-taxonomic relation extraction.

Reads the chat-format data from scripts/prepare_s5_finetune.py
(data/s5_ft/{train,val}.jsonl: {"messages": [system, user, assistant]}) and trains a
4-bit QLoRA adapter. Loss is computed on the assistant turn only (the relations JSON),
so the model learns to MAP (types + passage) → relations, including abstaining
({"triples": []}) for relation-free docs.

Run on a GPU node. See the README for the full fine-tuning and serving workflow.

Qwen3 is a hybrid-thinking model: training renders the chat template with
enable_thinking=False (direct JSON output, no <think> block), so eval/serving must also
use enable_thinking=False (debug_s5 --no-thinking).

Example:
    python scripts/finetune_s5.py \
        --model Qwen/Qwen3-14B \
        --train data/s5_ft/train.jsonl --val data/s5_ft/val.jsonl \
        --output-dir models/s5_qlora
"""
from __future__ import annotations

import argparse
import os

# Reduce fragmentation OOMs (the trainer flagged large reserved-but-unallocated memory).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-14B",
                    help="Base model: HF id or local path (default: cached Qwen3-14B)")
    ap.add_argument("--train", default="data/s5_ft/train.jsonl")
    ap.add_argument("--val", default=None, help="Optional; unused in training (eval is off)")
    ap.add_argument("--output-dir", default="models/s5_qlora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1, help="per-device train batch size")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--no-4bit", action="store_true", help="Full LoRA (no 4-bit) if you have the VRAM")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, torch_dtype=torch.bfloat16, device_map="auto",
    )

    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # val is optional (whole-data scope has no held-out split); in-training eval is off
    # anyway, so it's only loaded if provided.
    data_files = {"train": args.train}
    if args.val:
        data_files["val"] = args.val
    ds = load_dataset("json", data_files=data_files)

    # Pre-render to plain text with thinking DISABLED. Qwen3 is a hybrid-thinking model;
    # we want direct JSON output (no <think> block), so train and infer both with
    # enable_thinking=False. (Harmless for non-thinking templates — the kwarg is ignored.)
    def render(ex):
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False,
                                                enable_thinking=False)}
    ds = ds.map(render, remove_columns=ds["train"].column_names)

    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, save_strategy="epoch",
        eval_strategy="no",   # in-training eval OOMs on long few-shot seqs (batch-8 spike)
                              # and eval_loss is empty-dominated/uninformative — use debug_s5 instead.
        per_device_eval_batch_size=1,
        bf16=True, max_length=args.max_seq_len, packing=False,
        gradient_checkpointing=True,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=cfg,
        train_dataset=ds["train"],
        peft_config=peft_cfg, processing_class=tok,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"\nSaved LoRA adapter → {args.output_dir}")


if __name__ == "__main__":
    main()
