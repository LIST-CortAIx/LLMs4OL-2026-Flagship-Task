"""Step 0 — Embedding indices for S3 RAG, S1/S2 few-shot, and S4 taxonomy."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Retriever:
    """In-memory kNN retriever over embedded gold (term, type) pairs."""

    def __init__(
        self,
        pairs: list[dict],
        embeddings: "np.ndarray",  # noqa: F821
        model_name: str,
    ) -> None:
        self.pairs = pairs
        self.embeddings = embeddings
        self.model_name = model_name
        self._model = None
        self._pair_terms_norm: list[str] | None = None
        self._pair_token_sets: list[set] | None = None

    def warm(self) -> None:
        """Load the embedding model eagerly (call before parallel use)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    @staticmethod
    def _norm_term(s: str) -> str:
        import re
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

    def _build_lexical_index(self) -> None:
        """Precompute normalized term strings + token sets for lexical scoring."""
        self._pair_terms_norm = [self._norm_term(p["term"]) for p in self.pairs]
        self._pair_token_sets = [set(t.split()) for t in self._pair_terms_norm]

    def _lexical_scores(self, term: str):
        """Lexical similarity of `term` to every pair term: 1.0 exact, 0.7 substring,
        else token Jaccard. Returns a numpy vector aligned to self.pairs."""
        import numpy as np
        qn = self._norm_term(term)
        qtok = set(qn.split())
        out = np.zeros(len(self.pairs), dtype=np.float32)
        if not qn:
            return out
        for j, pn in enumerate(self._pair_terms_norm):  # type: ignore[arg-type]
            if pn == qn:
                out[j] = 1.0
            elif pn and (qn in pn or pn in qn):
                out[j] = 0.7
            else:
                ptok = self._pair_token_sets[j]  # type: ignore[index]
                if qtok and ptok:
                    inter = len(qtok & ptok)
                    if inter:
                        out[j] = 0.5 * inter / len(qtok | ptok)
        return out

    def query(self, query_text: str, k: int = 3) -> list[dict]:
        """Return the k most similar gold (term, types) pairs."""
        return self.batch_query([query_text], k=k)[0]

    def batch_query(self, query_texts: list[str], k: int = 3,
                    hybrid: bool = False, alpha: float = 0.5,
                    lex_keys: list[str] | None = None) -> list[list[dict]]:
        """Return top-k similar pairs for each query text in one encode call.

        When ``hybrid`` is set, the embedding cosine score is combined with a
        lexical score (``alpha`` weight) computed from ``lex_keys`` (the bare
        term, defaulting to ``query_texts``) — boosting training pairs whose
        term matches the query term exactly / by substring / by token overlap.
        """
        import numpy as np

        self.warm()
        q_embs = self._model.encode(  # type: ignore[union-attr]
            query_texts, normalize_embeddings=True, show_progress_bar=False
        )
        if hybrid:
            if self._pair_terms_norm is None:
                self._build_lexical_index()
            keys = lex_keys if lex_keys is not None else query_texts
        results = []
        for i, q_emb in enumerate(q_embs):
            scores = self.embeddings @ q_emb
            if hybrid:
                scores = scores + alpha * self._lexical_scores(keys[i])
            top_k_idx = np.argsort(scores)[::-1][:k]
            results.append([self.pairs[i2] for i2 in top_k_idx])
        return results


def build(
    gold_path: str | Path,
    index_dir: str | Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 512,
    _model=None,
) -> None:
    """Build and save an embedding index over gold (term, type) pairs."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    gold_path = Path(gold_path)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    with open(gold_path) as f:
        data = json.load(f)
    pairs: list[dict] = data["term_typings"]
    logger.info(
        "S0: building retriever index over %d gold pairs, model=%s",
        len(pairs), model_name,
    )

    texts = [p["term"] for p in pairs]
    model = _model or SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    np.save(index_dir / "embeddings.npy", embeddings)
    (index_dir / "pairs.json").write_text(
        json.dumps(pairs, ensure_ascii=False), encoding="utf-8"
    )
    (index_dir / "model.txt").write_text(model_name, encoding="utf-8")

    logger.info(
        "S0: index saved to %s (%d pairs, dim=%d)",
        index_dir, len(pairs), embeddings.shape[1],
    )


class DocRetriever:
    """Embedding-based kNN retriever over training documents for S1/S2 few-shot."""

    def __init__(
        self,
        docs: list[dict],
        embeddings: "np.ndarray",  # noqa: F821
        model_name: str,
        type_embeddings: "np.ndarray | None" = None,  # noqa: F821
    ) -> None:
        self.docs = docs
        self.embeddings = embeddings
        self.type_embeddings = type_embeddings
        self.model_name = model_name
        self._model = None
        from threading import Lock
        self._model_lock = Lock()

    def warm(self) -> None:
        """Load the embedding model eagerly (call before parallel use)."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self.model_name)

    def query(self, text: str, k: int = 3,
              required_field: str | None = None) -> list[dict]:
        """Return similar training docs, optionally requiring a non-empty field."""
        import numpy as np
        self.warm()
        q_emb = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False)
        scores = self.embeddings @ q_emb[0]
        ranked_idx = np.argsort(scores)[::-1]
        results: list[dict] = []
        for idx in ranked_idx:
            doc = self.docs[int(idx)]
            if required_field is not None and not doc.get(required_field):
                continue
            results.append(doc)
            if len(results) == k:
                break
        return results

    def query_by_type_labels(self, text: str, k: int = 3,
                              required_field: str | None = None) -> list[dict]:
        """Return training docs whose gold type labels best match the text."""
        import numpy as np
        if self.type_embeddings is None:
            raise RuntimeError(
                "Type-label index missing — rebuild with doc_type_embeddings.npy.")
        self.warm()
        q_emb = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False)
        scores = self.type_embeddings @ q_emb[0]
        ranked_idx = np.argsort(scores)[::-1]
        results: list[dict] = []
        for idx in ranked_idx:
            doc = self.docs[int(idx)]
            if required_field is not None and not doc.get(required_field):
                continue
            results.append(doc)
            if len(results) == k:
                break
        return results

    def query_hybrid_score(self, text: str, k: int = 3,
                           required_field: str | None = None,
                           doc_weight: float = 0.5,
                           type_weight: float = 0.5) -> list[dict]:
        """Rank docs by weighted sum of doc-text and type-label similarity."""
        import numpy as np
        if self.type_embeddings is None:
            raise RuntimeError("Type-label index missing.")
        self.warm()
        q_emb = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False)
        scores = doc_weight * (self.embeddings @ q_emb[0]) + \
                 type_weight * (self.type_embeddings @ q_emb[0])
        ranked_idx = np.argsort(scores)[::-1]
        results: list[dict] = []
        for idx in ranked_idx:
            doc = self.docs[int(idx)]
            if required_field is not None and not doc.get(required_field):
                continue
            results.append(doc)
            if len(results) == k:
                break
        return results

    def query_doc_pool_by_type_labels(self, text: str, k: int = 40,
                                       pool_size: int = 100,
                                       required_field: str | None = None) -> list[dict]:
        """Doc-similarity pool, then re-ranked by type-label score."""
        import numpy as np
        if self.type_embeddings is None:
            raise RuntimeError("Type-label index missing.")
        self.warm()
        q_emb = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False)
        doc_scores  = self.embeddings @ q_emb[0]
        type_scores = self.type_embeddings @ q_emb[0]
        pool_idx: list[int] = []
        for idx in np.argsort(doc_scores)[::-1]:
            doc = self.docs[int(idx)]
            if required_field is not None and not doc.get(required_field):
                continue
            pool_idx.append(int(idx))
            if len(pool_idx) == pool_size:
                break
        pool_positions = sorted(
            range(len(pool_idx)),
            key=lambda p: (-type_scores[pool_idx[p]], -doc_scores[pool_idx[p]], p))
        return [self.docs[pool_idx[p]] for p in pool_positions[:k]]

    def query_representative(self, text: str, k: int = 20,
                              pool_size: int = 100,
                              required_field: str | None = None) -> list[dict]:
        """Most central examples from a similar-doc pool (max pairwise similarity sum)."""
        import numpy as np
        self.warm()
        q_emb = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False)
        scores = self.embeddings @ q_emb[0]
        cand_idx: list[int] = []
        for idx in np.argsort(scores)[::-1]:
            doc = self.docs[int(idx)]
            if required_field is not None and not doc.get(required_field):
                continue
            cand_idx.append(int(idx))
            if len(cand_idx) == pool_size:
                break
        if len(cand_idx) <= k:
            return [self.docs[i] for i in cand_idx]
        cand_emb = self.embeddings[cand_idx]
        centrality = (cand_emb @ cand_emb.T).sum(axis=0)
        positions = sorted(range(len(cand_idx)),
                           key=lambda p: (-centrality[p], -scores[cand_idx[p]], p))[:k]
        return [self.docs[cand_idx[p]] for p in positions]

    def query_representative_diverse(self, text: str, k: int = 20,
                                      pool_size: int = 100,
                                      required_field: str | None = None) -> list[dict]:
        """Half central, half diverse examples from a similar-doc pool."""
        import numpy as np
        self.warm()
        q_emb = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False)
        scores = self.embeddings @ q_emb[0]
        cand_idx: list[int] = []
        for idx in np.argsort(scores)[::-1]:
            doc = self.docs[int(idx)]
            if required_field is not None and not doc.get(required_field):
                continue
            cand_idx.append(int(idx))
            if len(cand_idx) == pool_size:
                break
        if len(cand_idx) <= k:
            return [self.docs[i] for i in cand_idx]
        cand_emb = self.embeddings[cand_idx]
        centrality = (cand_emb @ cand_emb.T).sum(axis=0)
        rep_k = k // 2
        div_k = k - rep_k
        rep_pos = sorted(range(len(cand_idx)),
                         key=lambda p: (-centrality[p], -scores[cand_idx[p]], p))[:rep_k]
        selected = set(rep_pos)
        div_pos: list[int] = []
        for p in sorted(range(len(cand_idx)),
                        key=lambda p: (centrality[p], -scores[cand_idx[p]], p)):
            if p not in selected:
                div_pos.append(p)
                selected.add(p)
                if len(div_pos) == div_k:
                    break
        return [self.docs[cand_idx[p]] for p in rep_pos + div_pos]

    def sample(self, k: int = 3, required_field: str | None = None,
               seed: int | None = None) -> list[dict]:
        """Return k random training docs, optionally requiring a non-empty field."""
        import random
        candidates = [d for d in self.docs
                      if required_field is None or d.get(required_field)]
        rng = random.Random(seed)
        return rng.sample(candidates, min(k, len(candidates)))


def build_doc_index(
    training_path: str | Path,
    index_dir: str | Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 512,
    correct_terms: bool = False,
    _model=None,
) -> None:
    """Build embedding indices over training documents for S1/S2 few-shot.

    Produces two embeddings in ``index_dir``:
    - ``doc_embeddings.npy``       — full document text (text mode)
    - ``doc_embeddings_terms.npy`` — concatenated gold terms (terms mode)

    Both are built in one pass so a single ``run_step.py s0`` call covers
    both retrieval modes.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    training_path = Path(training_path)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    with open(training_path) as f:
        training_data = json.load(f)

    if correct_terms:
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).resolve().parent))
        from prepare_data import correct_term as _correct_term

    docs: list[dict] = []
    for sample in training_data:
        triples = sample.get("primitive-ontology-triples", [])
        text = sample["context"]
        raw_terms = {t[0] for t in triples if len(t) == 3 and t[1] == "instance-of"}
        taxonomic_pairs = [
            {"parent": t[2], "child": t[0]}
            for t in triples
            if len(t) == 3 and t[1] == "is-a"
        ]
        # Collect types from instance-of (objects) AND is-a (both heads and tails)
        types = {t[2] for t in triples if len(t) == 3 and t[1] == "instance-of"}
        types.update(
            label for t in triples if len(t) == 3 and t[1] == "is-a"
            for label in (t[0], t[2])
        )
        if correct_terms:
            terms = {_correct_term(t, text) for t in raw_terms}
        else:
            terms = raw_terms
        docs.append({
            "id": sample["id"],
            "text": sample["context"],
            "terms": sorted(terms),
            "types": sorted(types),
            "taxonomic_pairs": taxonomic_pairs,
        })

    model = _model or SentenceTransformer(model_name)

    # ── Text mode (full document text) ───────────────────────────────────────
    logger.info("S0 DocRetriever: encoding %d docs (text mode), model=%s",
                len(docs), model_name)
    text_embs = model.encode(
        [d["text"] for d in docs],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    np.save(index_dir / "doc_embeddings.npy", text_embs)

    # ── Terms mode (concatenated gold terms) ─────────────────────────────────
    # Docs with no terms use their text as fallback so they remain queryable.
    logger.info("S0 DocRetriever: encoding %d docs (terms mode)", len(docs))
    terms_texts = [
        " ".join(d["terms"]) if d["terms"] else d["text"]
        for d in docs
    ]
    terms_embs = model.encode(
        terms_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    np.save(index_dir / "doc_embeddings_terms.npy", terms_embs)

    # ── Type-label mode (gold types serialized for S2 retrieval) ─────────────
    logger.info("S0 DocRetriever: encoding %d docs (type-label mode)", len(docs))
    type_texts = [
        " ; ".join(d["types"]) if d["types"] else d["text"]
        for d in docs
    ]
    types_embs = model.encode(
        type_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    np.save(index_dir / "doc_type_embeddings.npy", types_embs)

    (index_dir / "doc_training.json").write_text(
        json.dumps(docs, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "S0 DocRetriever: saved to %s (%d docs, dim=%d, both modes)",
        index_dir, len(docs), text_embs.shape[1],
    )


def load_doc_index(index_dir: str | Path, retriever_mode: str = "text") -> DocRetriever:
    """Load a previously built DocRetriever from disk.

    Args:
        index_dir: Directory produced by :func:`build_doc_index`.
        retriever_mode: Embedding mode:
            - ``"text"`` (default) — full document text
            - ``"terms"`` — concatenated gold terms (for S1 retrieval)
            - ``"types"`` — concatenated gold types (for S2 retrieval)
    """
    import numpy as np

    index_dir = Path(index_dir)
    emb_file = {
        "terms": "doc_embeddings_terms.npy",
        "types": "doc_embeddings_types.npy",
    }.get(retriever_mode, "doc_embeddings.npy")
    embeddings = np.load(index_dir / emb_file)
    docs = json.loads((index_dir / "doc_training.json").read_text(encoding="utf-8"))
    model_name = (index_dir / "model.txt").read_text(encoding="utf-8").strip()
    logger.info(
        "S0 DocRetriever: loaded — %d training docs, dim=%d, model=%s, mode=%s",
        len(docs), embeddings.shape[1], model_name, retriever_mode,
    )
    # Load type embeddings if available (for S2 type-similarity modes)
    # Support both names (doc_type_embeddings.npy and doc_embeddings_types.npy)
    type_emb_path = index_dir / "doc_type_embeddings.npy"
    if not type_emb_path.exists():
        type_emb_path = index_dir / "doc_embeddings_types.npy"
    type_embeddings = np.load(type_emb_path) if type_emb_path.exists() else None

    return DocRetriever(docs=docs, embeddings=embeddings, model_name=model_name,
                        type_embeddings=type_embeddings)


def load(index_dir: str | Path) -> Retriever:
    """Load a previously built Retriever from disk."""
    import numpy as np

    index_dir = Path(index_dir)
    embeddings = np.load(index_dir / "embeddings.npy")
    pairs = json.loads((index_dir / "pairs.json").read_text(encoding="utf-8"))
    model_name = (index_dir / "model.txt").read_text(encoding="utf-8").strip()

    logger.info(
        "S0: loaded retriever index — %d pairs, dim=%d, model=%s",
        len(pairs), embeddings.shape[1], model_name,
    )
    return Retriever(pairs=pairs, embeddings=embeddings, model_name=model_name)


# ---------------------------------------------------------------------------
# Taxonomy Parent Index (S4 embedding approach)
# ---------------------------------------------------------------------------

class TaxonomyParentIndex:
    """kNN retriever over training is-a parent types for S4 embedding approach."""

    def __init__(
        self,
        parents: list[str],
        embeddings: "np.ndarray",  # noqa: F821
        model_name: str,
    ) -> None:
        self.parents = parents
        self.embeddings = embeddings
        self.model_name = model_name
        self._model = None

    def batch_query(
        self, child_types: list[str], k: int = 1
    ) -> list[list[tuple[str, float]]]:
        """Return top-k (parent, cosine_similarity) pairs for each child type."""
        import numpy as np

        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

        q_embs = self._model.encode(
            child_types, normalize_embeddings=True, show_progress_bar=True
        )
        results = []
        for q_emb in q_embs:
            scores = self.embeddings @ q_emb
            top_k_idx = np.argsort(scores)[::-1][:k]
            results.append([(self.parents[i], float(scores[i])) for i in top_k_idx])
        return results


def build_taxonomy_parent_index(
    training_path: str | Path,
    index_dir: str | Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 512,
    _model=None,
) -> None:
    """Build an embedding index over unique is-a parent types from training data."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    training_path = Path(training_path)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    with open(training_path) as f:
        training_data = json.load(f)

    parents: set[str] = set()
    for sample in training_data:
        for triple in sample.get("primitive-ontology-triples", []):
            if len(triple) == 3 and triple[1] == "is-a":
                parents.add(triple[2])

    parents_list = sorted(parents)
    logger.info(
        "S0 TaxonomyParentIndex: encoding %d unique is-a parents, model=%s",
        len(parents_list), model_name,
    )

    model = _model or SentenceTransformer(model_name)
    embeddings = model.encode(
        parents_list,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    np.save(index_dir / "taxonomy_parent_embeddings.npy", embeddings)
    (index_dir / "taxonomy_parents.json").write_text(
        json.dumps(parents_list, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "S0 TaxonomyParentIndex: saved to %s (%d parents, dim=%d)",
        index_dir, len(parents_list), embeddings.shape[1],
    )


def load_taxonomy_parent_index(index_dir: str | Path) -> TaxonomyParentIndex:
    """Load a previously built TaxonomyParentIndex from disk."""
    import numpy as np

    index_dir = Path(index_dir)
    embeddings = np.load(index_dir / "taxonomy_parent_embeddings.npy")
    parents = json.loads(
        (index_dir / "taxonomy_parents.json").read_text(encoding="utf-8")
    )
    model_name = (index_dir / "model.txt").read_text(encoding="utf-8").strip()
    logger.info(
        "S0 TaxonomyParentIndex: loaded — %d parents, dim=%d, model=%s",
        len(parents), embeddings.shape[1], model_name,
    )
    return TaxonomyParentIndex(parents=parents, embeddings=embeddings, model_name=model_name)
