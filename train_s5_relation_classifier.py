"""Train a binary classifier: does a document have any NON-TAXONOMIC relation?

S5 relations are every triple except is-a and instance-of. Only ~7% of documents
have any (61/859 in the split; ~303/3444 in train) — the other ~93% must produce an
empty S5 output. This classifier gates S5: documents predicted relation-free skip the
LLM entirely (S1 R4 analog), keeping the extraction prompt untouched for the rest.

High recall on the positive class is prioritised — missing a relation-bearing doc
loses all its relations silently.

Usage:
    # development (threshold tuned on the split test):
    python train_s5_relation_classifier.py
    # final submission (threshold tuned by CV on the whole training set):
    python train_s5_relation_classifier.py --whole --output models/s5_relation_classifier_full.pkl
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

NON_REL = {"is-a", "instance-of"}


def has_relation(sample: dict) -> bool:
    """True if the doc has any non-taxonomic relation (not is-a / instance-of)."""
    for t in sample.get("primitive-ontology-triples", []):
        if len(t) == 3 and t[1] not in NON_REL:
            return True
    return False


def load_labels(data: list[dict]) -> tuple[list[str], list[int]]:
    texts = [d["context"] for d in data]
    labels = [1 if has_relation(d) else 0 for d in data]
    return texts, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train",     default="data/train_task_a.json")
    parser.add_argument("--test-data", default="data/splits/test.json")
    parser.add_argument("--test-gold", default="data/splits/test_gold/s5_gold.json")
    parser.add_argument("--output",    default="outputs/s5_relation_classifier.pkl")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (default: optimise for recall≥0.95)")
    parser.add_argument("--whole", action="store_true",
                        help="Tune threshold by CV on the whole training set (ignores --test-*)")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    print("Loading training data …")
    train_data = json.loads(Path(args.train).read_text())
    train_texts, train_labels = load_labels(train_data)
    pos = sum(train_labels); neg = len(train_labels) - pos
    print(f"Train: {pos} pos / {neg} neg  ({100*pos/len(train_labels):.1f}% pos)")

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=np.array(train_labels))
    class_weight = {0: cw[0], 1: cw[1]}

    def make_pipeline():
        return Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50_000,
                                      sublinear_tf=True, min_df=2)),
            ("clf", LogisticRegression(class_weight=class_weight, max_iter=1000,
                                       C=1.0, solver="lbfgs")),
        ])

    pipeline = make_pipeline()

    if args.whole:
        print(f"\nTuning threshold via {args.cv_folds}-fold CV on the whole training data …")
        proba = cross_val_predict(make_pipeline(), train_texts, train_labels,
                                  cv=args.cv_folds, method="predict_proba", n_jobs=-1)[:, 1]
        eval_labels = train_labels
        print("Fitting final pipeline on all training data …")
        pipeline.fit(train_texts, train_labels)
        where = "out-of-fold CV"
    else:
        test_data = json.loads(Path(args.test_data).read_text())
        rel_ids = {d for r in json.loads(Path(args.test_gold).read_text())["non_taxonomic_relations"]
                   for d in r.get("source_doc_ids", [])}
        test_texts = [d["context"] for d in test_data]
        eval_labels = [1 if d["id"] in rel_ids else 0 for d in test_data]
        print(f"Test : {sum(eval_labels)} pos / {len(eval_labels)-sum(eval_labels)} neg")
        print("\nTraining …")
        pipeline.fit(train_texts, train_labels)
        proba = pipeline.predict_proba(test_texts)[:, 1]
        where = "split test"

    precisions, recalls, thresholds = precision_recall_curve(eval_labels, proba)
    if args.threshold is not None:
        threshold = args.threshold
    else:
        valid = [(p, r, t) for p, r, t in zip(precisions, recalls, thresholds) if r >= 0.95]
        threshold = max(valid, key=lambda x: x[0])[2] if valid else thresholds[np.argmax(recalls)]

    preds = (proba >= threshold).astype(int)
    print(f"\nThreshold: {threshold:.3f}  (evaluated on {where})")
    print(classification_report(eval_labels, preds,
                                target_names=["no-relation", "has-relation"], digits=3))
    n_calls = int(sum(preds)); tp = int(sum(1 for p, l in zip(preds, eval_labels) if p == 1 and l == 1))
    fn = int(sum(1 for p, l in zip(preds, eval_labels) if p == 0 and l == 1))
    print(f"LLM calls needed : {n_calls}/{len(preds)} ({100*n_calls/len(preds):.1f}%)")
    print(f"Relation docs    : {tp}/{sum(eval_labels)} (recall={tp/sum(eval_labels):.3f})")
    print(f"Missed (silent)  : {fn}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({"pipeline": pipeline, "threshold": threshold}, f)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
