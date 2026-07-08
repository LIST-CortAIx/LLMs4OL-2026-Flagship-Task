#!/usr/bin/env python3
"""Statistics and charts for the LLMs4OL Task A training data.

Usage:
    python scripts/analyse_training_data.py
    python scripts/analyse_training_data.py --data data/train_task_a.json --out outputs/stats
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


# Relations that map to each pipeline step
REL_TERM_TYPING = "instance-of"
REL_TAXONOMY    = "is-a"


# ---------------------------------------------------------------------------
# Data loading and stats computation
# ---------------------------------------------------------------------------

def load(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def compute(data: list[dict], tokenizer=None) -> dict:
    s: dict = {}

    context_len, context_tokens, triples_per_doc = [], [], []
    all_triples: list[tuple[str, str, str]] = []

    texts = [doc["context"] for doc in data]

    if tokenizer is not None:
        print("Tokenizing contexts …", flush=True)
        encoded = tokenizer(texts, add_special_tokens=False)
        token_counts = [len(ids) for ids in encoded["input_ids"]]
        s["token_source"] = getattr(tokenizer, "name_or_path", "provided tokenizer")
    else:
        token_counts = [len(t) // 4 for t in texts]
        s["token_source"] = "chars / 4 (estimate)"

    for doc, tok_count in zip(data, token_counts):
        words = doc["context"].split()
        context_len.append(len(words))
        context_tokens.append(tok_count)
        triples = [tuple(t) for t in doc.get("primitive-ontology-triples", []) if len(t) == 3]
        triples_per_doc.append(len(triples))
        all_triples.extend(triples)

    s["n_docs"]          = len(data)
    s["n_docs_empty"]    = sum(1 for c in triples_per_doc if c == 0)
    s["n_triples"]       = len(all_triples)
    s["context_len"]     = context_len
    s["context_tokens"]  = context_tokens
    s["triples_per_doc"] = triples_per_doc
    s["relation_counts"] = Counter(t[1] for t in all_triples)

    # --- instance-of: term typing (S3) ---
    io_pairs = [(t[0], t[2]) for t in all_triples if t[1] == REL_TERM_TYPING]
    s["n_io_pairs"]     = len(io_pairs)
    s["n_io_dedup"]     = len(set(io_pairs))          # unique (term, type) pairs
    s["n_unique_terms"] = len({p[0] for p in io_pairs})
    # Docs that have at least one instance-of triple (can be evaluated for S1)
    docs_with_io = sum(
        1 for doc in data
        if any(len(t) == 3 and t[1] == REL_TERM_TYPING
               for t in doc.get("primitive-ontology-triples", []))
    )
    s["n_docs_with_io"] = docs_with_io
    term_types: dict[str, set] = defaultdict(set)
    for term, typ in io_pairs:
        term_types[term].add(typ)
    s["types_per_term"] = [len(v) for v in term_types.values()]
    s["type_freq"]      = Counter(p[1] for p in io_pairs)

    # --- is-a: taxonomy (S4) ---
    isa_pairs = [(t[0], t[2]) for t in all_triples if t[1] == REL_TAXONOMY]
    s["n_isa_pairs"] = len(isa_pairs)
    s["n_isa_dedup"] = len(set(isa_pairs))            # unique (child, parent) pairs
    children_of: dict[str, set] = defaultdict(set)
    parents_of:  dict[str, set] = defaultdict(set)
    for child, parent in isa_pairs:
        children_of[parent].add(child)
        parents_of[child].add(parent)
    s["subtypes_per_type"] = [len(v) for v in children_of.values()]
    s["parents_per_type"]  = [len(v) for v in parents_of.values()]

    # S2: all ontological types = both sides of is-a + instance-of targets
    isa_nodes    = {p[0] for p in isa_pairs} | {p[1] for p in isa_pairs}
    io_targets   = {p[1] for p in io_pairs}
    all_type_nodes = isa_nodes | io_targets
    s["n_unique_types"]   = len(all_type_nodes)
    s["n_types_from_isa"] = len(isa_nodes)
    s["n_types_io_only"]  = len(io_targets - isa_nodes)

    # Which is-a types appear verbatim in at least one document's context?
    type_in_text: dict[str, bool] = {}
    for doc in data:
        ctx = doc["context"].lower()
        for triple in doc.get("primitive-ontology-triples", []):
            if len(triple) != 3 or triple[1] != REL_TAXONOMY:
                continue
            for node in (triple[0], triple[2]):
                if node not in type_in_text:
                    type_in_text[node] = node.lower() in ctx
                elif not type_in_text[node]:
                    type_in_text[node] = node.lower() in ctx
    s["n_types_in_text"]       = sum(1 for v in type_in_text.values() if v)
    s["n_types_never_in_text"] = sum(1 for v in type_in_text.values() if not v)

    # --- non-taxonomic (S5) ---
    skip = {REL_TERM_TYPING, REL_TAXONOMY}
    non_tax = [(t[0], t[1], t[2]) for t in all_triples if t[1] not in skip]
    s["n_non_tax"]          = len(non_tax)
    s["n_non_tax_dedup"]    = len(set(non_tax))
    s["non_tax_rel_counts"] = Counter(t[1] for t in non_tax)

    return s


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "—"

def _hist_summary(vals: list[int]) -> str:
    if not vals:
        return "—"
    sv = sorted(vals)
    return (f"min={sv[0]}  median={sv[len(sv)//2]}  "
            f"max={sv[-1]}  mean={sum(sv)/len(sv):.1f}")


def report(s: dict) -> None:
    sep = "─" * 62
    print(sep)
    print("  LLMs4OL Task A — Training Data Statistics")
    print(sep)

    print(f"\n{'Documents':<35} {s['n_docs']:>8,}")
    print(f"  with at least one triple          {s['n_docs'] - s['n_docs_empty']:>8,}"
          f"  ({_pct(s['n_docs'] - s['n_docs_empty'], s['n_docs'])})")
    print(f"  empty                             {s['n_docs_empty']:>8,}"
          f"  ({_pct(s['n_docs_empty'], s['n_docs'])})")
    print(f"{'Total triples':<35} {s['n_triples']:>8,}")

    print(f"\n{'Step':<6} {'Relation':<20} {'Triples':>9}  {'Share':>6}")
    print("  " + "·" * 46)
    for rel, cnt in sorted(s["relation_counts"].items(), key=lambda x: -x[1]):
        step = ("S3" if rel == REL_TERM_TYPING
                else "S4" if rel == REL_TAXONOMY
                else "S5")
        print(f"  {step:<6}{rel:<24}{cnt:>7,}  {_pct(cnt, s['n_triples']):>6}")

    print(f"\n{'Context length (words)':<35} {_hist_summary(s['context_len'])}")
    print(f"{'Context tokens [{src}]'.format(src=s['token_source']):<35} {_hist_summary(s['context_tokens'])}")
    n = s["n_docs"]
    tokens = s["context_tokens"]
    for threshold in (512, 1024, 2048):
        under = sum(1 for t in tokens if t < threshold)
        print(f"  < {threshold} tokens{'':<24} {under:>6,} / {n:,}  ({_pct(under, n)})")
    print(f"{'Triples per document':<35} {_hist_summary(s['triples_per_doc'])}")

    print(f"\n── S1 Term Extraction ────────────────────────────────────")
    print(f"  Docs with ≥1 instance-of triple   {s['n_docs_with_io']:>8,}"
          f"  ({_pct(s['n_docs_with_io'], s['n_docs'])}) ← S1 gold coverage")
    print(f"  Docs without instance-of          {s['n_docs'] - s['n_docs_with_io']:>8,}"
          f"  ({_pct(s['n_docs'] - s['n_docs_with_io'], s['n_docs'])}) ← only types (S4/S5)")
    print(f"\n── S3 Term Typing (instance-of) ──────────────────────────")
    print(f"  Unique terms                      {s['n_unique_terms']:>8,}")
    print(f"  Unique (term, type) pairs (dedup) {s['n_io_dedup']:>8,}")
    print(f"  (term, type) pairs (with dups)    {s['n_io_pairs']:>8,}")
    print(f"  Types per term  {_hist_summary(s['types_per_term'])}")
    print(f"\n── S2/S4 Type Vocabulary & Taxonomy ─────────────────────")
    print(f"  Unique types (S2 vocabulary)      {s['n_unique_types']:>8,}")
    print(f"    from is-a taxonomy nodes        {s['n_types_from_isa']:>8,}  ({_pct(s['n_types_from_isa'], s['n_unique_types'])})")
    print(f"    from instance-of targets only   {s['n_types_io_only']:>8,}  ({_pct(s['n_types_io_only'], s['n_unique_types'])})")
    n_isa = s["n_types_from_isa"]
    print(f"  Is-a types found in context text  {s['n_types_in_text']:>8,}  ({_pct(s['n_types_in_text'], n_isa)} of is-a types)")
    print(f"  Is-a types never in any context   {s['n_types_never_in_text']:>8,}  ({_pct(s['n_types_never_in_text'], n_isa)} of is-a types)")
    print(f"  Unique (child, parent) pairs      {s['n_isa_dedup']:>8,}")
    print(f"  Subtypes per type  {_hist_summary(s['subtypes_per_type'])}")
    print(f"  Parents per type   {_hist_summary(s['parents_per_type'])}")
    print(f"\n── S5 Non-Taxonomic ──────────────────────────────────────")
    print(f"  Unique non-taxonomic triples      {s['n_non_tax_dedup']:>8,}")
    for rel, cnt in s["non_tax_rel_counts"].most_common():
        print(f"    {rel:<36} {cnt:>5,}  ({_pct(cnt, s['n_non_tax'])})")
    print()


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _ax_hist(ax, vals: list[int], *, bins, color: str, xlabel: str, title: str,
             logscale: bool = False) -> None:
    mean = sum(vals) / len(vals)
    ax.hist(vals, bins=bins, color=color, edgecolor="white", linewidth=0.4)
    ax.axvline(mean, color="#c0392b", linestyle="--", linewidth=1.2,
               label=f"mean = {mean:.1f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("# documents" if "doc" in xlabel.lower() else "count")
    ax.set_title(title)
    ax.legend(fontsize=9)
    if logscale:
        ax.set_yscale("log")


def plot(s: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    out_dir.mkdir(parents=True, exist_ok=True)
    COLORS = {
        "blue":   "#2980b9",
        "orange": "#e67e22",
        "green":  "#27ae60",
        "purple": "#8e44ad",
        "red":    "#c0392b",
        "grey":   "#95a5a6",
    }
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })

    # ── Figure 1: Overview (2 × 2) ────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("LLMs4OL Task A — Training Data Overview  (n = {:,} documents)".format(
        s["n_docs"]), fontsize=12, fontweight="bold", y=1.01)

    # 1a — Relation type distribution
    ax = axes[0, 0]
    rel_items = sorted(s["relation_counts"].items(), key=lambda x: -x[1])
    labels = [r for r, _ in rel_items]
    counts = [c for _, c in rel_items]
    bar_colors = [
        COLORS["orange"] if l == REL_TERM_TYPING else
        COLORS["blue"]   if l == REL_TAXONOMY else
        COLORS["grey"]
        for l in labels
    ]
    bars = ax.bar(range(len(labels)), counts, color=bar_colors, width=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Relation Type Distribution")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.4,
                f"{cnt:,}", ha="center", va="bottom", fontsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=COLORS["orange"], label=f"{REL_TERM_TYPING} (S3)"),
        Patch(color=COLORS["blue"],   label=f"{REL_TAXONOMY} (S4)"),
        Patch(color=COLORS["grey"],   label="non-taxonomic (S5)"),
    ], fontsize=8)

    # 1b — Triples per document
    _ax_hist(axes[0, 1], s["triples_per_doc"],
             bins=50, color=COLORS["blue"],
             xlabel="Triples per document",
             title="Triples per Document")

    # 1c — Context length
    _ax_hist(axes[1, 0], s["context_len"],
             bins=60, color=COLORS["green"],
             xlabel="Context length (words)",
             title="Context Length Distribution")

    # 1d — Step scale comparison (bar chart, deduplicated counts)
    ax = axes[1, 1]
    step_labels = ["S1\nTerms", "S2\nTypes", "S3\nTerm\ntypings", "S4\nIs-a\npairs", "S5\nNon-tax"]
    step_counts = [
        s["n_unique_terms"],
        s["n_unique_types"],
        s["n_io_dedup"],
        s["n_isa_dedup"],
        s["n_non_tax_dedup"],
    ]
    step_colors = [COLORS["green"], COLORS["purple"], COLORS["orange"],
                   COLORS["blue"], COLORS["grey"]]
    bars = ax.bar(step_labels, step_counts, color=step_colors, width=0.6)
    ax.set_ylabel("Count")
    ax.set_title("Scale per Pipeline Step")
    for bar, cnt in zip(bars, step_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{cnt:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
    ))

    fig.tight_layout()
    p = out_dir / "overview.png"
    fig.savefig(p, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close(fig)

    # ── Figure 2: Term Typing & Taxonomy ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("S3 Term Typing & S4 Taxonomy — Complexity Analysis",
                 fontsize=12, fontweight="bold")

    # 2a — Types per term
    ax = axes[0]
    tpt = s["types_per_term"]
    max_bin = min(max(tpt) + 2, 12)
    ax.hist(tpt, bins=range(1, max_bin), color=COLORS["orange"],
            edgecolor="white", linewidth=0.4, align="left")
    ax.set_xlabel("Types assigned per term")
    ax.set_ylabel("# Terms")
    ax.set_title("Types per Term (S3)")
    ax.set_xticks(range(1, max_bin - 1))
    mean_tpt = sum(tpt) / len(tpt)
    ax.axvline(mean_tpt, color=COLORS["red"], linestyle="--",
               label=f"mean = {mean_tpt:.2f}")
    ax.legend(fontsize=9)

    # 2b — Taxonomy fan-out (subtypes per type)
    ax = axes[1]
    spt = s["subtypes_per_type"]
    max_bin = min(max(spt) + 2, 30)
    ax.hist(spt, bins=range(1, max_bin), color=COLORS["blue"],
            edgecolor="white", linewidth=0.4, align="left")
    ax.set_xlabel("Subtypes per type (fan-out)")
    ax.set_ylabel("# Types")
    ax.set_title("Taxonomy Fan-out (S4)")
    ax.set_yscale("log")
    mean_spt = sum(spt) / len(spt)
    ax.axvline(mean_spt, color=COLORS["red"], linestyle="--",
               label=f"mean = {mean_spt:.1f}")
    ax.legend(fontsize=9)

    # 2c — Parents per type (in-degree)
    ax = axes[2]
    ppt = s["parents_per_type"]
    max_bin = min(max(ppt) + 2, 12)
    ax.hist(ppt, bins=range(1, max_bin), color=COLORS["purple"],
            edgecolor="white", linewidth=0.4, align="left")
    ax.set_xlabel("Parents per type (in-degree)")
    ax.set_ylabel("# Types")
    ax.set_title("Taxonomy In-degree (S4)")
    ax.set_xticks(range(1, max_bin - 1))
    ax.set_yscale("log")
    mean_ppt = sum(ppt) / len(ppt)
    ax.axvline(mean_ppt, color=COLORS["red"], linestyle="--",
               label=f"mean = {mean_ppt:.2f}")
    ax.legend(fontsize=9)

    fig.tight_layout()
    p = out_dir / "typing_taxonomy.png"
    fig.savefig(p, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close(fig)

    # ── Figure 3: Top types & non-taxonomic relations ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Vocabulary & Non-Taxonomic Relations",
                 fontsize=12, fontweight="bold")

    # 3a — Top 20 types
    ax = axes[0]
    top_n = 20
    top_types = s["type_freq"].most_common(top_n)
    t_labels = [t[0] for t in top_types][::-1]
    t_counts = [t[1] for t in top_types][::-1]
    ax.barh(range(len(t_labels)), t_counts, color=COLORS["orange"], height=0.7)
    ax.set_yticks(range(len(t_labels)))
    ax.set_yticklabels(t_labels, fontsize=8)
    ax.set_xlabel("# term typings")
    ax.set_title(f"Top {top_n} Most Common Types (S2/S3)")
    for i, cnt in enumerate(t_counts):
        ax.text(cnt + 0.3, i, str(cnt), va="center", fontsize=8)

    # 3b — Non-taxonomic relation distribution
    ax = axes[1]
    non_tax_items = s["non_tax_rel_counts"].most_common()
    n_labels = [r for r, _ in non_tax_items]
    n_counts = [c for _, c in non_tax_items]
    x = range(len(n_labels))
    bars = ax.bar(x, n_counts, color=COLORS["grey"], width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(n_labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Non-Taxonomic Relation Distribution (S5)")
    for bar, cnt in zip(bars, n_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(cnt), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    p = out_dir / "vocabulary_and_s5.png"
    fig.savefig(p, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/train_task_a.json",
                        help="Path to train_task_a.json (default: data/train_task_a.json)")
    parser.add_argument("--out",  default="outputs/stats",
                        help="Directory to save charts (default: outputs/stats)")
    parser.add_argument("--no-charts", action="store_true",
                        help="Print statistics only, skip chart generation")
    parser.add_argument("--tokenizer", default=None,
                        help="HuggingFace tokenizer name/path for exact token counts "
                             "(e.g. Qwen/Qwen3-8B). Defaults to chars/4 estimate.")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"Data file not found: {data_path}")

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    data  = load(data_path)
    stats = compute(data, tokenizer=tokenizer)
    report(stats)

    if not args.no_charts:
        try:
            plot(stats, Path(args.out))
        except ImportError:
            print("matplotlib not available — skipping charts (pip install matplotlib)")


if __name__ == "__main__":
    main()
