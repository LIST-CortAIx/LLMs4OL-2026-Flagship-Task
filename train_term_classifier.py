"""Train a binary classifier to predict whether a document contains terms.

Trains on train_task_a.json and evaluates on data/splits/test.json.
Saves the fitted pipeline to outputs/term_classifier.pkl.

High recall on the positive class is prioritised — missing a term-bearing
document (false negative) is worse than running the LLM on a type-only doc
(false positive).

Usage:
    python train_term_classifier.py
    python train_term_classifier.py --threshold 0.3
"""

from __future__ import annotations
import argparse, json, pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight


def load_labels(data: list[dict]) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    for d in data:
        texts.append(d["context"])
        has_io = any(len(t) == 3 and t[1] == "instance-of"
                     for t in d.get("primitive-ontology-triples", []))
        labels.append(1 if has_io else 0)
    return texts, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",     default="data/train_task_a.json")
    parser.add_argument("--test-data", default="data/splits/test.json")
    parser.add_argument("--test-gold", default="data/splits/test_gold/s1_gold.json")
    parser.add_argument("--output",    default="outputs/term_classifier.pkl")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (default: optimise for recall≥0.95)")
    parser.add_argument("--whole", action="store_true",
                        help="Production mode: train AND tune the threshold on the whole "
                             "training data via cross-validation (ignores --test-data/--test-gold). "
                             "Use for the final submission classifier.")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of CV folds for threshold tuning in --whole mode")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading training data …")
    train_data  = json.loads(Path(args.train).read_text())
    train_texts, train_labels = load_labels(train_data)

    pos = sum(train_labels); neg = len(train_labels) - pos
    print(f"Train: {pos} pos / {neg} neg  ({100*pos/len(train_labels):.1f}% pos)")

    # ── Build pipeline ─────────────────────────────────────────────────────────
    cw = compute_class_weight("balanced", classes=np.array([0, 1]),
                              y=np.array(train_labels))
    class_weight = {0: cw[0], 1: cw[1]}

    def make_pipeline():
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=50_000,
                sublinear_tf=True,
                min_df=2,
            )),
            ("clf", LogisticRegression(
                class_weight=class_weight,
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
            )),
        ])

    pipeline = make_pipeline()

    if args.whole:
        # ── Whole-data mode: tune threshold via cross-validation, fit on all data ─
        print(f"\nTuning threshold via {args.cv_folds}-fold CV on the whole training data …")
        oof_proba = cross_val_predict(
            make_pipeline(), train_texts, train_labels,
            cv=args.cv_folds, method="predict_proba", n_jobs=-1,
        )[:, 1]
        eval_labels, proba = train_labels, oof_proba
        print("Fitting final pipeline on all training data …")
        pipeline.fit(train_texts, train_labels)
    else:
        # ── Split mode (dev): tune threshold on the held-out split test set ───────
        test_data  = json.loads(Path(args.test_data).read_text())
        gold_ids   = {did for t in json.loads(Path(args.test_gold).read_text())["terms"]
                      for did in t["source_doc_ids"]}
        test_texts  = [d["context"] for d in test_data]
        test_labels = [1 if d["id"] in gold_ids else 0 for d in test_data]
        print(f"Test : {sum(test_labels)} pos / {len(test_labels)-sum(test_labels)} neg")
        print("\nTraining …")
        pipeline.fit(train_texts, train_labels)
        proba = pipeline.predict_proba(test_texts)[:, 1]
        eval_labels = test_labels

    # ── Pick threshold (recall ≥ 0.95, maximise precision) ─────────────────────
    precisions, recalls, thresholds = precision_recall_curve(eval_labels, proba)
    if args.threshold is not None:
        threshold = args.threshold
    else:
        valid = [(p, r, t) for p, r, t in zip(precisions, recalls, thresholds)
                 if r >= 0.95]
        if valid:
            threshold = max(valid, key=lambda x: x[0])[2]
        else:
            threshold = thresholds[np.argmax(recalls)]

    preds = (proba >= threshold).astype(int)

    where = "out-of-fold CV" if args.whole else "split test"
    print(f"\nThreshold: {threshold:.3f}  (evaluated on {where})")
    print(classification_report(eval_labels, preds,
                                 target_names=["type-only", "has-terms"],
                                 digits=3))

    # LLM call savings
    n_calls = sum(preds)
    n_total = len(preds)
    tp = sum(1 for p, l in zip(preds, eval_labels) if p==1 and l==1)
    fn = sum(1 for p, l in zip(preds, eval_labels) if p==0 and l==1)
    print(f"LLM calls needed : {n_calls}/{n_total} ({100*n_calls/n_total:.1f}%)")
    print(f"Terms docs found : {tp}/{sum(eval_labels)} (recall={tp/sum(eval_labels):.3f})")
    print(f"Missed term docs : {fn}  (false negatives — silent misses)")

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({"pipeline": pipeline, "threshold": threshold}, f)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
