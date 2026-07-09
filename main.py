"""Portable Option 2 entry point for the LLMs4OL Flagship pipeline.

This script mirrors the orchestration in ``run_pipeline.slurm``:
S1 -> S2 -> S3 -> S4 -> S5 -> assemble.
It does not change the step implementations or their default configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _cfg_value(key: str, fallback: str) -> str:
    config = ROOT / "config.yaml"
    if not config.exists():
        return fallback
    for line in config.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        name, _, value = stripped.partition(":")
        if name.strip() == key:
            return value.split("#")[0].strip() or fallback
    return fallback


def _convert_test_input(test_input: str, corpus_output: str) -> None:
    src = ROOT / test_input if not Path(test_input).is_absolute() else Path(test_input)
    dst = ROOT / corpus_output if not Path(corpus_output).is_absolute() else Path(corpus_output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(src.read_text(encoding="utf-8"))
    with dst.open("w", encoding="utf-8") as handle:
        for item in data:
            handle.write(
                json.dumps({"id": item["id"], "text": item["context"]}, ensure_ascii=False)
                + "\n"
            )
    print(f"Converted {src} -> {dst} ({len(data)} documents)")


def _run(cmd: list[str], env: dict[str, str]) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"\n$ {printable}", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _fmt_time(seconds: int) -> str:
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m {seconds % 60:02d}s"


def _backend_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    for flag, value in (
        ("--base-url", args.base_url),
        ("--model", args.model),
        ("--api-key", args.api_key),
        ("--profile", args.profile),
        ("--reasoning-effort", args.reasoning_effort),
    ):
        if value:
            out.extend([flag, value])
    if args.insecure:
        out.append("--insecure")
    return out


def _s5_args(args: argparse.Namespace, backend_args: list[str]) -> list[str]:
    if args.s5_base_url:
        out = [
            "--base-url",
            args.s5_base_url,
            "--model",
            args.s5_model,
            "--profile",
            args.s5_profile,
            "--no-thinking",
            "--no-domain-inference",
            "--relation-classifier",
            "",
        ]
        if args.api_key:
            out.extend(["--api-key", args.api_key])
    else:
        out = list(backend_args)
    if args.s5_gate_threshold is not None:
        out.extend(["--gate-threshold", str(args.s5_gate_threshold)])
    return out


def _check_s4_index(index_dir: str) -> None:
    path = ROOT / index_dir if not Path(index_dir).is_absolute() else Path(index_dir)
    doc_training = path / "doc_training.json"
    doc_embeddings = path / "doc_embeddings.npy"
    if not doc_training.exists() or not doc_embeddings.exists():
        raise SystemExit(
            f"ERROR: S4 document index missing in {path}.\n"
            "Build indexes first with: python prepare_data.py --scope whole"
        )
    docs = json.loads(doc_training.read_text(encoding="utf-8"))
    if not any(doc.get("taxonomic_pairs") for doc in docs):
        raise SystemExit(
            "ERROR: S4 index has no taxonomic_pairs metadata; rebuild it with "
            "python prepare_data.py --scope whole"
        )


def _eval_step(
    step_cmd: list[str],
    step: str,
    prediction: str,
    gold: str,
    enabled: bool,
    env: dict[str, str],
) -> None:
    if not enabled:
        return
    gold_path = ROOT / gold if not Path(gold).is_absolute() else Path(gold)
    if not gold_path.exists():
        print(f"WARN: gold not found ({gold}) - skipping eval for {step}")
        return
    _run(step_cmd + ["eval", "--step", step, "--predictions", prediction, "--gold", gold], env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full LLMs4OL pipeline: S1 -> S2 -> S3 -> S4 -> S5 -> assemble."
    )
    parser.add_argument("--scope", choices=["whole", "split"], default=os.environ.get("SCOPE", "whole"))
    parser.add_argument("--test-input", default="data/test_task_a_input.json")
    parser.add_argument("--input", default=None, help="Corpus JSONL. If omitted, --test-input is converted.")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--output", default=None)
    parser.add_argument("--index-dir", default=None, help="Force one S0 index dir for S1, S2, S3 and S5.")
    parser.add_argument("--s4-index", default=None, help="S0 index used by S4 few-shot retrieval.")
    parser.add_argument("--workers", default=None)
    parser.add_argument("--gold-dir", default=None)
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument("--eval", dest="do_eval", action="store_true", default=None)
    eval_group.add_argument("--no-eval", dest="do_eval", action="store_false")

    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY"))
    parser.add_argument("--profile", default=os.environ.get("PIPELINE_PROFILE"))
    parser.add_argument("--insecure", action="store_true", default=os.environ.get("PIPELINE_INSECURE") == "1")
    parser.add_argument("--reasoning-effort", default=os.environ.get("PIPELINE_REASONING_EFFORT"))
    parser.add_argument("--s5-base-url", default=os.environ.get("S5_BASE_URL"))
    parser.add_argument("--s5-model", default=os.environ.get("S5_MODEL", "s5-ft"))
    parser.add_argument("--s5-profile", default=os.environ.get("S5_PROFILE", "qwen"))
    parser.add_argument("--s5-gate-threshold", type=float, default=None)
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Do not force TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE=1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.scope == "split":
        args.input = args.input or "data/splits/test_corpus.jsonl"
        args.index_dir = args.index_dir or "indexes/split_s0_retriever"
        args.gold_dir = args.gold_dir or "data/splits/test_gold"
        do_eval = True if args.do_eval is None else args.do_eval
    else:
        do_eval = False if args.do_eval is None else args.do_eval
        args.gold_dir = args.gold_dir or "data/gold"

    out_dir = args.out_dir
    output = args.output or f"{out_dir}/submission.json"
    workers = args.workers or _cfg_value("workers", "8")
    input_path = args.input or f"{out_dir}/test_corpus.jsonl"
    s4_index = args.s4_index or args.index_dir or "indexes/s0_retriever_full"

    env = os.environ.copy()
    env["SCOPE"] = args.scope
    if not args.allow_downloads:
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("HF_HUB_OFFLINE", "1")

    Path(ROOT / out_dir).mkdir(parents=True, exist_ok=True)
    if args.input is None:
        _convert_test_input(args.test_input, input_path)

    _check_s4_index(s4_index)

    step = [sys.executable, str(ROOT / "run_step.py")]
    backend = _backend_args(args)
    s5_backend = _s5_args(args, backend)
    index_args = ["--index-dir", args.index_dir] if args.index_dir else []

    print("=" * 60)
    print(f"Scope       : {args.scope}")
    print(f"Input       : {input_path}")
    print(f"Output      : {output}")
    print(f"Index S1-5  : {args.index_dir or 'per-step config defaults'}")
    print(f"Index S4    : {s4_index}")
    print(f"Workers     : {workers}")
    print(f"Eval        : {'on' if do_eval else 'off'}")
    print("=" * 60)

    timings: dict[str, int] = {}
    t_pipeline = time.monotonic()

    def timed(name: str, command: list[str]) -> None:
        started = time.monotonic()
        _run(command, env)
        timings[name] = int(time.monotonic() - started)
        print(f"--- {name.upper()}: {_fmt_time(timings[name])} ---")

    timed(
        "s1",
        step
        + ["s1", "--input", input_path]
        + index_args
        + backend
        + ["--output", f"{out_dir}/s1_output.json", "--workers", str(workers)],
    )
    _eval_step(step, "s1", f"{out_dir}/s1_output.json", f"{args.gold_dir}/s1_gold.json", do_eval, env)

    timed(
        "s2",
        step
        + ["s2", "--input", input_path]
        + index_args
        + backend
        + ["--no-thinking", "--output", f"{out_dir}/s2_output.json", "--workers", str(workers)],
    )
    _eval_step(step, "s2", f"{out_dir}/s2_output.json", f"{args.gold_dir}/s2_gold.json", do_eval, env)

    timed(
        "s3",
        step
        + [
            "s3",
            "--s1-input",
            f"{out_dir}/s1_output.json",
            "--s2-input",
            f"{out_dir}/s2_output.json",
            "--corpus",
            input_path,
        ]
        + index_args
        + backend
        + ["--output", f"{out_dir}/s3_output.json", "--workers", str(workers)],
    )
    _eval_step(step, "s3", f"{out_dir}/s3_output.json", f"{args.gold_dir}/s3_gold.json", do_eval, env)

    timed(
        "s4",
        step
        + [
            "s4",
            "--s2-input",
            f"{out_dir}/s2_output.json",
            "--corpus",
            input_path,
            "--index-dir",
            s4_index,
            "--output",
            f"{out_dir}/s4_output.json",
            "--workers",
            str(workers),
            "--no-thinking",
        ]
        + backend,
    )
    _eval_step(step, "s4", f"{out_dir}/s4_output.json", f"{args.gold_dir}/s4_gold.json", do_eval, env)

    timed(
        "s5",
        step
        + [
            "s5",
            "--corpus",
            input_path,
            "--s1-input",
            f"{out_dir}/s1_output.json",
            "--s2-input",
            f"{out_dir}/s2_output.json",
            "--input",
            f"{out_dir}/s3_output.json",
        ]
        + s5_backend
        + ["--output", f"{out_dir}/s5_output.json", "--workers", str(workers)],
    )
    _eval_step(step, "s5", f"{out_dir}/s5_output.json", f"{args.gold_dir}/s5_gold.json", do_eval, env)

    _run(
        step
        + [
            "assemble",
            "--corpus",
            input_path,
            "--s1-input",
            f"{out_dir}/s1_output.json",
            "--s3-input",
            f"{out_dir}/s3_output.json",
            "--s4-input",
            f"{out_dir}/s4_output.json",
            "--s5-input",
            f"{out_dir}/s5_output.json",
            "--output",
            output,
        ],
        env,
    )
    _eval_step(step, "submission", output, f"{args.gold_dir}/submission_gold.json", do_eval, env)

    total = int(time.monotonic() - t_pipeline)
    print("\nTiming summary")
    for name in ("s1", "s2", "s3", "s4", "s5"):
        print(f"  {name:<6} {_fmt_time(timings[name])}")
    print(f"  {'TOTAL':<6} {_fmt_time(total)}")
    print(f"\nFinal submission written to {output}")


if __name__ == "__main__":
    main()
