"""Prompt templates for each pipeline step."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from models import ExtractedTerm


_S1_PER_DOC_SYSTEM = """\
You are an expert ontology engineer performing term extraction.
Given a single document, extract the key domain-specific terms — named individuals \
that would appear as subjects of "instance-of" triples in an ontology.

Return a JSON object with this exact schema:
{"terms": ["term1", "term2", ...]}

Rules:
- Terms are specific named entities, NOT abstract categories or generic nouns.
- Always use the exact form as it appears in the document text, lowercase. \
The examples below are for guidance on which entities to extract — \
their output format may differ from the text; always follow the text form.
- Be selective: only entities the document specifically defines or characterises.
- Exclude stopwords, generic verbs, overly abstract words ("thing", "entity").
- If nothing qualifies, return {"terms": []}."""

_S1_PER_DOC_USER = """\
{examples}Document:
{passage}"""

_S1_PER_DOC_SYSTEM_MIXED = """\
You are an expert ontology engineer performing term extraction.
Given a single document, extract specific named individuals — entities that would \
appear as subjects of "instance-of" triples in an ontology.

Return a JSON object with this exact schema:
{"terms": ["term1", "term2", ...]}

Rules:
- Terms are specific named entities, NOT abstract categories or generic nouns.
- Always use the exact form as it appears in the document text, lowercase. \
The examples below are for guidance — their output format may differ from the text; \
always follow the text form.
- Be selective: only entities the document specifically defines or characterises.
- Exclude stopwords, generic verbs, overly abstract words ("thing", "entity").
- The examples include documents WITH terms and documents WITHOUT terms (type-only \
ontology descriptions with no specific instances). Use them to judge which case \
applies to the current document.
- If nothing qualifies, return {"terms": []}."""

_S1_PER_DOC_SYSTEM_COT = """\
You are an expert ontology engineer performing term extraction.
Given a single document, reason step by step before producing the final answer:

Step 1 — DOMAIN: Identify the scientific/technical domain of the document in one sentence.
Step 2 — NATURE: Decide whether this document describes a type hierarchy (abstract \
categories and is-a relations) or contains specific named instances. If it is purely \
a type hierarchy with no specific individuals, output {"terms": []} immediately.
Step 3 — CANDIDATES: List all noun phrases that name specific entities in this domain.
Step 4 — FILTER: For each candidate, decide: is it a specific named individual (an instance \
that would appear as the subject of an "instance-of" triple)? Remove abstract categories, \
generic nouns, and common words.
Step 5 — NORMALIZE: Write each kept entity in the exact form it appears in the text, lowercase.
Step 6 — OUTPUT: Return the final list as a JSON object.

Return a JSON object with this exact schema:
{"terms": ["term1", "term2", ...]}

Rules:
- Always use the exact form as it appears in the document text, lowercase. \
The examples below are for guidance on which entities to extract — \
their output format may differ from the text; always follow the text form.
- Be selective: only entities the document specifically defines or characterises.
- Exclude stopwords, generic verbs, overly abstract words ("thing", "entity").
- Many documents describe only type hierarchies with no specific instances — \
returning {"terms": []} is correct for those.
- If nothing qualifies, return {"terms": []}."""

_S2_PER_DOC_SYSTEM = """\
You are an expert ontology engineer performing type extraction.
Given a single document, identify the abstract ontological types (class labels) \
that domain entities in the document are instances of — the Y in "X is-a Y" statements.

Return a JSON object with this exact schema:
{"types": ["type1", "type2", ...]}

Rules:
- Types are abstract, reusable categories (e.g. "sensor", "control device"), \
NOT specific instances like "temperature sensor".
- Canonical (singular, uninflected) form, lowercase.
- Prefer coarser types over fine-grained ones — broader is more reusable.
- Exclude overly generic labels ("thing", "entity", "concept", "object").
- If nothing qualifies, return {"types": []}.

Before producing the final answer, reason internally using the following procedure:

Step 1 — Understand the domain: read the text and infer the general domain or situation.
Step 2 — Detect candidate concepts: identify important nouns, noun phrases, actors, \
objects, events, processes, places, roles, and abstract notions mentioned or implied.
Step 3 — Decide whether each candidate is a class: keep it if it answers "What kind \
of thing is this?" and could have multiple instances. Reject specific individuals, \
literal values, attributes, or relationships.
Step 4 — Generalize when appropriate: if the text mentions a specific entity, infer \
the corresponding class (e.g. "Marie Curie" → "scientist", "Paris" → "city").
Step 5 — Separate classes from properties/values: "patient" is a class; "age" is an \
attribute; "treated by" is a relation; "urgent" is a value unless the domain requires \
a class such as "priority level".
Step 6 — Normalize and merge: use singular lowercase form. Merge synonyms or \
near-duplicates. Avoid overly generic classes unless important in this domain."""

_S2_PER_DOC_USER = """\
{examples}Document:
{passage}"""

_S2_PER_DOC_SYSTEM_COT = """\
You are an expert ontology engineer performing type extraction.
Given a single document, identify the abstract ontological types (class labels) \
that domain entities in the document are instances of.

Return a JSON object with this exact schema:
{"types": ["type1", "type2", ...]}

Rules:
- Types are abstract, reusable categories (e.g. "sensor", "control device"), \
NOT specific instances like "temperature sensor".
- Canonical (singular, uninflected) form, lowercase.
- Prefer coarser types over fine-grained ones — broader is more reusable.
- Exclude overly generic labels ("thing", "entity", "concept", "object").
- If nothing qualifies, return {"types": []}.


Before producing the final answer, reason internally using the following extraction procedure:

Step 1 — Understand the domain
Read the text and infer the general domain or situation being described. Keep this context in mind when deciding which concepts are ontology-relevant.

Step 2 — Detect candidate concepts
Identify important nouns, noun phrases, actors, objects, events, processes, places, documents, roles, organizations, and abstract notions mentioned or strongly implied in the text.

Step 3 — Decide whether each candidate is a class
For each candidate, ask whether it represents a general category that could have multiple instances.
Keep it as a class if it answers the question: “What kind of thing is this?”
Reject it if it is only a specific individual, a literal value, an attribute, or a relationship.

Step 4 — Generalize when appropriate
If the text mentions a specific entity, infer the corresponding class only when useful.
For example:
- “Marie Curie” may suggest the class “Scientist”
- “Paris” may suggest the class “City”
- “Visa card” may suggest “CreditCard” or “PaymentCard”

Step 5 — Separate classes from properties and values
Do not confuse entities with their attributes or relations.
For example:
- “Patient” is a class
- “age” is usually an attribute
- “treated by” is a relationship
- “urgent” is usually a value, unless the domain requires a class such as “PriorityLevel”

Step 6 — Normalize and merge
Normalize each class name using singular PascalCase.
Merge synonyms or near-duplicates.
Avoid overly generic classes unless they are important in the domain.
"""



def build_s1_per_doc_messages(
    passage: str,
    examples: list[dict] | None = None,
    examples_header: str = "Examples from similar documents:",
    cot: bool = False,
    examples_empty: list[dict] | None = None,
    examples_empty_header: str = "Examples of type-only documents (no terms):",
) -> list[dict[str, str]]:
    """Build messages for S1: per-document term extraction with RAG few-shot.

    Args:
        passage: Single document text.
        examples: Training docs with terms (retrieved by DocRetriever).
        cot: Use Chain-of-Thought system prompt.
        examples_empty: Training docs with no terms — shown to calibrate
            the model for type-only documents. Activates the mixed prompt.
        examples_empty_header: Header for the empty-term example block.
    """
    # Build with-term examples block
    ex_str = ""
    if examples:
        lines = [examples_header + "\n"]
        for ex in examples:
            lines.append(ex["text"].strip())
            lines.append(json.dumps({"terms": ex.get("terms", [])}) + "\n")
        ex_str = "\n".join(lines) + "\n"

    # Build without-term examples block (mixed mode)
    ex_empty_str = ""
    if examples_empty:
        lines = [examples_empty_header + "\n"]
        for ex in examples_empty:
            lines.append(ex["text"].strip())
            lines.append(json.dumps({"terms": []}) + "\n")
        ex_empty_str = "\n".join(lines) + "\n"

    # Select system prompt — mixed is the default; CoT uses its own updated version
    if cot:
        system = _S1_PER_DOC_SYSTEM_COT
    else:
        system = _S1_PER_DOC_SYSTEM_MIXED

    combined_examples = ex_str + ex_empty_str
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": _S1_PER_DOC_USER.format(
                examples=combined_examples,
                passage=passage.strip(),
            ),
        },
    ]


def build_s2_per_doc_messages(
    passage: str,
    examples: list[dict] | None = None,
    examples_header: str = "Examples from similar documents:",
    cot: bool = False,
) -> list[dict[str, str]]:
    """Build messages for S2: per-document type extraction with RAG few-shot.

    Args:
        passage: Single document text.
        examples: Training docs retrieved by DocRetriever — each has "text" and "types".
    """
    ex_str = ""
    if examples:
        lines = [examples_header + "\n"]
        for i, ex in enumerate(examples):
            snippet = ex["text"].strip()
            types_str = ", ".join(ex.get("types", [])) or "(none)"
            lines.append(f"[Example {i}] {snippet}")
            lines.append(f"  → types: {types_str}\n")
        ex_str = "\n".join(lines) + "\n"

    system = _S2_PER_DOC_SYSTEM_COT if cot else _S2_PER_DOC_SYSTEM
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": _S2_PER_DOC_USER.format(
                examples=ex_str,
                passage=passage.strip(),
            ),
        },
    ]


_S3_SYSTEM = """\
You are an expert ontology engineer performing term typing.
Given a list of terms and the known type vocabulary for a domain, assign \
one or more types from the vocabulary to each term.

Return a JSON object with this exact schema:
{
  "results": [
    {
      "term_index": <int>,
      "term": "<term>",
      "types": ["Type1", "Type2", ...]
    }
  ]
}

Rules:
- Include one entry per term in the same order as the input.
- Only assign types from the provided vocabulary; do NOT invent new types.
- Assign all types that genuinely apply (multiple inheritance is allowed).
- If no type fits, use an empty list."""

_S3_USER = """\
Type vocabulary:
{type_vocab}

Assign types to the following {n} term(s):

{terms}"""


def build_s3_messages(
    terms: list[str],
    type_vocab: list[str],
    contexts: list[str] | None = None,
    examples: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Build messages for S3: term typing.

    Args:
        terms: Terms to type (one batch).
        type_vocab: All valid type labels for the domain.
        contexts: Source sentences for each term, parallel to ``terms``.
            When provided, each term is rendered as
            ``[i] term (found in: "sentence")`` giving the model the
            grounding context it needs for accurate typing.
        examples: Optional few-shot examples as list of
            ``{"term": ..., "types": [...]}`` dicts.

    Returns:
        List of message dicts for ``LLMClient.chat_json()``.
    """
    vocab_str = ", ".join(f'"{t}"' for t in sorted(type_vocab))

    lines = []
    for i, term in enumerate(terms):
        ctx = (contexts[i] if contexts and i < len(contexts) else "").strip()
        if ctx:
            lines.append(f'[{i}] {term} (found in: "{ctx}")')
        else:
            lines.append(f"[{i}] {term}")
    numbered = "\n".join(lines)

    user_content = _S3_USER.format(
        type_vocab=vocab_str, n=len(terms), terms=numbered
    )
    if examples:
        ex_lines = "\n".join(
            f'  {{"term": "{e["term"]}", "types": {e["types"]}}}'
            for e in examples[:5]
        )
        user_content = f"Examples:\n{ex_lines}\n\n{user_content}"

    return [
        {"role": "system", "content": _S3_SYSTEM},
        {"role": "user", "content": user_content},
    ]


_S3_RAG_SYSTEM = """\
You are an expert ontology engineer performing term typing.
For each term you are given a few similar training examples showing the correct
ontological type at the appropriate level of specificity for this domain.
Assign the most specific type(s) to each term, following the granularity shown
in the examples.

Return a JSON object with this exact schema:
{
  "results": [
    {
      "term_index": <int>,
      "term": "<term>",
      "types": ["type1"]
    }
  ]
}

Rules:
- Include one entry per term, in the same order as the input.
- Types must be lowercase ontological class labels matching the style of the examples.
- Return 1–2 types maximum; prefer the single most specific type that applies.
- If none of the retrieved examples are relevant, infer the most specific type
  you can from the term itself and the domain context.
- Do NOT use generic super-types ("entity", "thing", "concept", "substance")
  unless the examples themselves use that level of generality."""

_S3_RAG_USER = """\
Assign ontological type(s) to the following {n} term(s).
For each term, use the similar training examples as guidance for specificity.

{items}"""


_S3_CONSTRAINED_SYSTEM = """\
You are an expert ontology engineer performing term typing.
For each term you are given its context and a CANDIDATE LIST of allowed types,
gathered from the most similar training terms.

Choose the best-matching type(s) for each term FROM ITS CANDIDATE LIST ONLY.
- Match the granularity used in the candidates — prefer the coarser option.
- Return 1 type when possible; at most 2.
- Pick the closest candidate even if the fit is imperfect; only return an empty
  list if no candidate is remotely related.

Return a JSON object with this exact schema:
{
  "results": [
    { "term_index": <int>, "term": "<term>", "types": ["<one of the candidates>"] }
  ]
}
Include one entry per term, in input order. Types must be lowercase and copied
verbatim from the candidate list."""

_S3_CONSTRAINED_USER = """\
Assign type(s) to the following {n} term(s) by choosing from each term's candidate list.

{items}"""


_S3_LENIENT_SYSTEM = """\
You are an expert ontology engineer performing term typing.
For each term you are given its context and a SUGGESTED LIST of types,
gathered from the most similar training terms.

Prefer a type FROM the suggested list when one genuinely fits the term — these
are calibrated to the right granularity (prefer the coarser option).
If NONE of the suggestions fits, assign the most accurate type you can infer
from the term and the document context — you are NOT restricted to the list.
- Return 1 type when possible; at most 2.

Return a JSON object with this exact schema:
{
  "results": [
    { "term_index": <int>, "term": "<term>", "types": ["type1"] }
  ]
}
Include one entry per term, in input order. Types must be lowercase."""

_S3_LENIENT_USER = """\
Assign type(s) to the following {n} term(s), using each term's suggested list when a suggestion fits.

{items}"""


def build_s3_constrained_messages(
    terms: "list[ExtractedTerm]",
    candidates: list[list[str]],
    contexts: list[str] | None = None,
    doc_context: str | None = None,
    context_chars: int = 150,
    lenient: bool = False,
    frequent_types=None,
) -> list[dict[str, str]]:
    """Build messages for S3 candidate-list typing.

    Args:
        terms: Terms to type (one batch).
        candidates: Parallel list — for each term, the type labels (typically the
            unique types of its k retrieved gold pairs).
        contexts: Optional per-term context strings (override context_sentence).
        doc_context: Optional document text prepended once (full-doc grounding).
        context_chars: Max characters of per-term context to show.
        lenient: If True, the list is presented as *suggestions* and the model may
            generate a type outside it when none fits (retrieve-then-select with an
            escape hatch). If False, strict closed-world (pick from the list only).
    """
    label = "Suggestions" if lenient else "Candidates"
    items: list[str] = []
    for i, (term, cands) in enumerate(zip(terms, candidates)):
        ctx = contexts[i] if contexts is not None else term.context_sentence
        lines = [f"[{i}] {term.text}"]
        if ctx:
            lines.append(f'    Context: "{ctx[:context_chars].strip()}"')
        cand_str = ", ".join(f'"{c}"' for c in cands) if cands else "(none retrieved)"
        lines.append(f"    {label}: {cand_str}")
        items.append("\n".join(lines))

    user_tmpl = _S3_LENIENT_USER if lenient else _S3_CONSTRAINED_USER
    user = user_tmpl.format(n=len(terms), items="\n\n".join(items))
    user = _frequent_block(frequent_types) + user
    if doc_context:
        user = (f"Document the terms were extracted from (for context):\n"
                f"\"\"\"\n{doc_context.strip()}\n\"\"\"\n\n" + user)
    return [
        {"role": "system", "content": _S3_LENIENT_SYSTEM if lenient else _S3_CONSTRAINED_SYSTEM},
        {"role": "user", "content": user},
    ]


def _frequent_block(frequent_types) -> str:
    """Frequent ontology types as a *fallback* tier, not a default preference.

    Priority order: the term's own similar examples first; the frequent list only
    when no example type fits; free inference otherwise. The explicit guard against
    replacing a fitting specific type with a general one prevents over-generalization.
    """
    if not frequent_types:
        return ""
    types = ", ".join(f'"{t}"' for t in frequent_types)
    return ("When choosing each term's type, follow this priority:\n"
            "1. Prefer a type shown in that term's similar examples when one fits.\n"
            f"2. Only if no example type fits, you MAY use one of the most frequent "
            f"ontology types when it genuinely fits: {types}.\n"
            "3. Otherwise, infer the most accurate type yourself.\n"
            "Never replace a specific type that already fits with a more general one.\n\n")


def build_s3_rag_messages(
    terms: "list[ExtractedTerm]",
    term_examples: list[list[dict]],
    contexts: list[str] | None = None,
    doc_context: str | None = None,
    context_chars: int = 150,
    frequent_types=None,
) -> list[dict[str, str]]:
    """Build messages for S3 RAG-based term typing.

    Each term is rendered with its optional context sentence and the k retrieved
    gold (term, type) pairs that are closest to it in embedding space.

    Args:
        terms: Terms to type (one batch), each carrying an optional
            ``context_sentence`` from S1.
        term_examples: Parallel list — for each term, the k retrieved gold
            records ``{"term": ..., "types": [...]}``.
        contexts: Optional per-term context strings (parallel to ``terms``) that
            override each term's ``context_sentence`` — used to inject the
            containing sentence and/or doc title at inference time.
        doc_context: Optional document text prepended once as a shared
            "Document" block (full-document grounding). Applies to the whole
            batch (all terms must belong to the same document).
        context_chars: Max characters of per-term context to show (default 150).

    Returns:
        List of message dicts for ``LLMClient.chat_json()``.
    """
    items: list[str] = []
    for i, (term, examples) in enumerate(zip(terms, term_examples)):
        ctx = contexts[i] if contexts is not None else term.context_sentence
        lines = [f"[{i}] {term.text}"]
        if ctx:
            snippet = ctx[:context_chars].strip()
            lines.append(f'    Context: "{snippet}"')
        if examples:
            lines.append("    Similar examples (term → expected output):")
            for ex in examples:
                lines.append(f'    {ex["term"]} → {json.dumps({"types": ex.get("types", [])})}')

        items.append("\n".join(lines))

    user = _S3_RAG_USER.format(n=len(terms), items="\n\n".join(items))
    user = _frequent_block(frequent_types) + user
    if doc_context:
        user = (f"Document the terms were extracted from (for context):\n"
                f"\"\"\"\n{doc_context.strip()}\n\"\"\"\n\n" + user)

    return [
        {"role": "system", "content": _S3_RAG_SYSTEM},
        {"role": "user", "content": user},
    ]


_S4_CLUSTER_SYSTEM = """\
You are an expert ontology engineer.
Given a list of semantic type labels from a domain ontology, group them into
semantically coherent clusters. Types in the same cluster should be candidates
for is-a (subsumption) relationships with each other.

Return a JSON object with this exact schema:
{{
  "clusters": {{
    "0": ["Type1", "Type2", ...],
    "1": ["Type3", "Type4", ...],
    ...
  }}
}}

Rules:
- Every type must appear in exactly one cluster.
- Cluster IDs are consecutive integers starting from 0.
- Aim for approximately {n_clusters} clusters; use fewer if the types naturally group into fewer.
- Group types that share a semantic domain or could stand in a subsumption relationship.
- Singleton clusters are allowed for types with no clear relatives."""

_S4_CLUSTER_USER = """\
Group the following {n} types into approximately {n_clusters} semantic clusters:

{types}"""


def build_s4_cluster_messages(
    types: list[str],
    n_clusters: int = 10,
) -> list[dict[str, str]]:
    """Build messages for S4 semantic clustering pre-step."""
    types_str = "\n".join(f"- {t}" for t in types)
    return [
        {
            "role": "system",
            "content": _S4_CLUSTER_SYSTEM.format(n_clusters=n_clusters),
        },
        {
            "role": "user",
            "content": _S4_CLUSTER_USER.format(
                n=len(types), n_clusters=n_clusters, types=types_str
            ),
        },
    ]


_S4_SYSTEM = """\
You are an expert ontology engineer performing taxonomy discovery.
For each concept pair (A, B) you are asked: "What is the relation between A and B?"

A valid taxonomic parent-child relation means:
every instance of B is necessarily an instance of A \
(strict subsumption: B ⊆ A).

Critically check the direction: if "every B is an A" is true but "every A is \
a B" is false, then A is the parent and B is the child.

Classify each pair into exactly one of these three labels:

1. TAXONOMIC_IS_A
Use this label only if A is a strict parent of B:
every B is an A, but NOT every A is a B.

2. RELATED_BUT_NOT_TAXONOMIC
Use this label if A and B are semantically or scientifically related, but A is \
not a strict parent of B.
This includes cases such as association, part-of, co-occurrence, thematic \
similarity, causal relation, mechanism relation, process-outcome relation, \
or domain-related relation.

3. INVALID_OR_UNCLEAR
Use this label if there is no clear relation between A and B, if the relation \
is too ambiguous, if the reverse taxonomic direction holds, or if both \
directions hold as equivalent concepts.

Return a JSON object with this exact schema:
{
  "results": [
    {
      "pair_index": <int>,
      "parent": "<concept>",
      "child": "<concept>",
      "relation_label": "TAXONOMIC_IS_A|RELATED_BUT_NOT_TAXONOMIC|INVALID_OR_UNCLEAR"
    }
  ]
}

Rules:
- Include one entry per pair in the same order as the input.
- relation_label=TAXONOMIC_IS_A means: every [child] is a [parent], but NOT \
every [parent] is a [child].
- If the reverse direction holds, mark INVALID_OR_UNCLEAR.
- If both directions hold, mark INVALID_OR_UNCLEAR because the concepts are \
equivalent, not parent-child.
- Be conservative: use TAXONOMIC_IS_A only when the subsumption is unambiguous.
"""

_S4_USER = """\
For each pair below, classify the relation with exactly one relation_label:
TAXONOMIC_IS_A, RELATED_BUT_NOT_TAXONOMIC, or INVALID_OR_UNCLEAR.
Use TAXONOMIC_IS_A only when every [child] is a [parent] by definition.
Use INVALID_OR_UNCLEAR if the reverse relationship is valid or if the concepts \
are equivalent.

{n} concept pair(s) to evaluate:

{pairs}"""


_S4_SINGLE_PARENT_RULE = """\

Additional comparative selection rule for the candidate pairs:
- Evaluate all candidate parents shown for the same child jointly.
- For each child, assign TAXONOMIC_IS_A to at most one candidate parent.
- If several candidate parents appear taxonomically valid, choose only the \
single parent you are most confident is correct for that child.
- Assign INVALID_OR_UNCLEAR to every other candidate parent for that child, \
even if another candidate could also be interpreted as a broader ancestor.
- If no candidate parent is clearly valid, do not assign TAXONOMIC_IS_A to \
any pair for that child.
- This single-parent rule applies to the candidate pairs to evaluate and takes \
precedence over any pattern in the few-shot examples.
"""


_S4_SINGLE_PARENT_USER_RULE = """\
Compare all candidate parents for each child. Return TAXONOMIC_IS_A for at \
most one parent per child: the single parent you are most confident is valid. \
Return INVALID_OR_UNCLEAR for every other parent candidate of that child.

"""


_S4_BOOLEAN_SYSTEM = """\
You are an expert ontology engineer performing taxonomy discovery.
For each candidate pair (parent, child), decide whether the parent-child \
is-a relation is valid.

A valid taxonomic relation means:
every instance of the child concept is necessarily an instance of the parent \
concept (strict subsumption), and the two concepts are not equivalent.

Return a JSON object with this exact schema:
{
  "results": [
    {
      "pair_index": <int>,
      "parent": "<concept>",
      "child": "<concept>",
      "is_parent": true|false
    }
  ]
}

Rules:
- Include one entry per pair in the same order as the input.
- Set is_parent=true only when every [child] is a [parent] by definition.
- If the reverse relation is true, set is_parent=false.
- If the concepts are equivalent, ambiguous, merely related, or unrelated, set \
is_parent=false.
- Be conservative: true only for unambiguous strict is-a relations."""


_S4_BOOLEAN_USER = """\
For each pair below, predict is_parent as true or false.

{n} concept pair(s) to evaluate:

{pairs}"""


def _format_s4_examples(
    examples: list[dict[str, object]] | None,
    *,
    boolean: bool,
    grouped: bool = False,
) -> str:
    if not examples:
        return ""

    positive = [e for e in examples if bool(e.get("is_parent"))]
    negative = [e for e in examples if not bool(e.get("is_parent"))]

    if grouped:
        positive_by_child: dict[str, list[str]] = {}
        negative_by_child: dict[str, list[str]] = {}
        for e in positive:
            parent = str(e.get("parent", "")).strip()
            child = str(e.get("child", "")).strip()
            if parent and child:
                positive_by_child.setdefault(child, []).append(parent)
        for e in negative:
            parent = str(e.get("parent", "")).strip()
            child = str(e.get("child", "")).strip()
            if parent and child:
                negative_by_child.setdefault(child, []).append(parent)

        lines = ["Contrastive examples from similar training documents:"]
        for child, parents in positive_by_child.items():
            positive_parts = [
                f"{child!r} is a {parent!r}"
                for parent in dict.fromkeys(parents)
            ]
            negative_parts = [
                f"{child!r} is not {parent!r}"
                for parent in dict.fromkeys(negative_by_child.get(child, []))
            ]
            statement = " and ".join(positive_parts)
            if negative_parts:
                statement = f"{statement}, but {', and '.join(negative_parts)}"
            lines.append(f"- {statement}.")
        return "\n".join(lines)

    def _lines(items: list[dict[str, object]], label: str) -> list[str]:
        lines = [label]
        for e in items:
            parent = str(e.get("parent", "")).strip()
            child = str(e.get("child", "")).strip()
            if not parent or not child:
                continue
            if boolean:
                value = "true" if bool(e.get("is_parent")) else "false"
                lines.append(
                    f"- parent={parent!r}, child={child!r} -> is_parent={value}"
                )
            else:
                relation = (
                    "TAXONOMIC_IS_A"
                    if bool(e.get("is_parent"))
                    else "RELATED_BUT_NOT_TAXONOMIC"
                )
                lines.append(
                    f"- parent={parent!r}, child={child!r} -> "
                    f"relation_label={relation}"
                )
        return lines

    sections: list[str] = ["Examples from similar training documents:"]
    if positive:
        sections.extend(_lines(positive, "Positive taxonomic examples:"))
    if negative:
        sections.extend(_lines(negative, "Negative non-taxonomic examples:"))
    return "\n".join(sections)


_S4_DEPTH_SYSTEM = """\
You are an expert ontology engineer refining a taxonomy hierarchy.
Given a parent concept and its currently known sub-types, identify any \
additional sub-types from a candidate list that are genuinely missing.

Return a JSON object with this exact schema:
{
  "additional_subtypes": ["type1", "type2", ...]
}

Rules:
- Only include concepts from the candidate list.
- Only include concepts with a genuine strict is-a (subsumption) relationship \
to the parent — every instance of the sub-type must be an instance of the parent.
- Return an empty list if nothing is missing.
- Do NOT repeat concepts already in the known sub-types list."""

_S4_DEPTH_USER = """\
Parent concept: "{parent}"

Already known sub-types of "{parent}":
{known}

Candidate concepts (select missing sub-types from this list only):
{candidates}"""


def build_s4_messages(
    pairs: list[tuple[str, str]],
    examples: list[dict[str, object]] | None = None,
    *,
    mode: str = "labels",
    example_style: str = "sequential",
    single_parent_per_child: bool = False,
) -> list[dict[str, str]]:
    """Build messages for S4: taxonomy discovery.

    Args:
        pairs: List of ``(candidate_parent, candidate_child)`` tuples.
        mode: ``"labels"`` for relation_label output or ``"boolean"`` for
            is_parent true/false output.
        examples: Optional few-shot examples with ``parent``, ``child``, and
            ``is_parent`` fields.
        example_style: ``"sequential"`` lists examples independently;
            ``"grouped"`` renders contrastive statements grouped by child.
        single_parent_per_child: Require at most one TAXONOMIC_IS_A parent for
            each child represented in the candidate pairs.

    Returns:
        List of message dicts for ``LLMClient.chat_json()``.
    """
    numbered = "\n".join(
        f"[{i}] Is every {c!r} a {p!r}? → parent={p!r}, child={c!r}"
        for i, (p, c) in enumerate(pairs)
    )
    if mode == "boolean":
        system = _S4_BOOLEAN_SYSTEM
        user_content = _S4_BOOLEAN_USER.format(n=len(pairs), pairs=numbered)
        examples_content = _format_s4_examples(
            examples,
            boolean=True,
            grouped=example_style == "grouped",
        )
    else:
        system = _S4_SYSTEM
        user_content = _S4_USER.format(n=len(pairs), pairs=numbered)
        examples_content = _format_s4_examples(
            examples,
            boolean=False,
            grouped=example_style == "grouped",
        )

    if single_parent_per_child:
        system = f"{system.rstrip()}\n{_S4_SINGLE_PARENT_RULE}"
        user_content = f"{_S4_SINGLE_PARENT_USER_RULE}{user_content}"

    if examples_content:
        user_content = f"{examples_content}\n\n{user_content}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def build_s4_depth_messages(
    parent: str,
    known_children: list[str],
    candidates: list[str],
) -> list[dict[str, str]]:
    """Build messages for S4 depth re-prompt: find missing sub-types of a parent."""
    known_str = "\n".join(f"- {c}" for c in known_children) if known_children else "(none yet)"
    candidates_str = "\n".join(f"- {c}" for c in candidates)
    return [
        {"role": "system", "content": _S4_DEPTH_SYSTEM},
        {
            "role": "user",
            "content": _S4_DEPTH_USER.format(
                parent=parent,
                known=known_str,
                candidates=candidates_str,
            ),
        },
    ]


_S5_DOMAIN_SYSTEM = """\
You are an expert ontology engineer.
Given a set of semantic type labels and sample passages from a corpus, \
infer the knowledge domain of the corpus.

Return a JSON object with this exact schema:
{
  "domain_name": "<short domain name, 1-3 words>",
  "domain_description": "<1-2 sentence description of the domain scope and typical terminology>"
}"""

_S5_DOMAIN_USER = """\
Semantic type vocabulary:
{types}

Sample passages:
{passages}"""


def build_s5_domain_messages(
    types: list[str],
    sample_passages: list[str],
) -> list[dict[str, str]]:
    """Build messages for the S5 domain inference pre-step."""
    types_str = ", ".join(types[:50])
    passages_str = "\n\n".join(
        f"[{i}] {p[:300]}" for i, p in enumerate(sample_passages[:5])
    )
    return [
        {"role": "system", "content": _S5_DOMAIN_SYSTEM},
        {
            "role": "user",
            "content": _S5_DOMAIN_USER.format(types=types_str, passages=passages_str),
        },
    ]


_S5_EXTRACT_SYSTEM = """\
You are an expert ontology engineer. You are given a document and the list of \
ontological TYPES (classes) it defines. Output the non-taxonomic relations that \
hold BETWEEN THESE TYPES — every relation except is-a (subclass) and instance-of \
(term typing).

Both the subject and the object of every relation MUST be types from the provided \
list. Relations hold between types, never between bare terms.

Find relations from two sources:
1. STATED IN TEXT — relations the passage explicitly states or strongly implies.
2. ONTOLOGY AXIOMS — relations that follow from the types' meaning even if unstated. \
Assert these ONLY when clearly warranted — do not guess; precision matters.

Use these exact labels when they apply (otherwise a short verb phrase from the text):
- "is defined by"  — a class formally defined in this ontology; written as a \
self-relation [X, is defined by, X]
- "type"           — a class declared as an individual; written as \
[X, type, "named individual"]
- "equivalent class" — synonymous or equivalent classes (also known as, same as); symmetric
- "disjoint with"  — mutually exclusive classes (opposite, never co-occurring); symmetric
- "has part" / "part_of" — composition (X has part Y / X is a component of Y)
- "exact match"    — identical concept across sources; often a self-relation [X, exact match, X]
- "database_cross_reference" — common name ↔ formal/scientific identifier or cross-ontology counterpart
- "see also"       — related but not formally equivalent
- "tree view"      — hierarchical structural nesting

Return a JSON object with this exact schema:
{
  "triples": [
    {"subject": "<type>", "relation": "<relation label or verb phrase>", "object": "<type>"}
  ]
}

Rules:
- NEVER output is-a or instance-of relations.
- Subject and object MUST appear in the type list above.
- Be precise: only assert a relation you are confident in. Return an empty triples \
list if no relation clearly holds."""

_S5_EXTRACT_SYSTEM_V2 = """\
You are an expert ontology engineer. You are given a document and the list of its \
ontological TYPES (classes). Use the document's CONTEXT to decide which types are \
related, and output the non-taxonomic relations that hold between them. Both the \
subject and object of every relation MUST be types from the list.

Prefer relations the passage STATES or STRONGLY IMPLIES:
- alternative names / abbreviations / variants of the same concept → "equivalent class" \
(e.g. "X and Y are both …", "Y, also called X", "X (abbrev. Y)") [symmetric]
- composition → "has part" / "part_of";  spatial containment → "located in"
- a class given a formal/scientific id or cross-ontology counterpart → "database_cross_reference"
- origin / development → "derives from" / "develops_from";  related but not equal → "see also"
- a class formally defined here → "is defined by" as a self-loop [X, is defined by, X]
- a class declared as an individual → [X, type, "named individual"]

You MAY also assert a structural ontology axiom that clearly follows from the classes' \
meaning, but be conservative:
- "equivalent class" for classes that denote the same concept (synonyms / aliases), \
whether explicitly stated or well established.
- "disjoint with" ONLY for two classes that are mutually exclusive; do NOT assert \
disjointness by default across a set of sibling / coordinate types (e.g. do not relate \
every pair of measurement units, climate categories, or scales).

Use one of these exact labels; only if none fits, a short verb phrase from the text.

Return a JSON object with this exact schema:
{
  "triples": [
    {"subject": "<type>", "relation": "<relation label or verb phrase>", "object": "<type>"}
  ]
}
Return an empty triples list if no relation clearly holds."""

_S5_EXTRACT_USER = """\
Types in this document: {doc_types}

Typed terms: {typed_terms}

Document:
{passage}"""


# Complete non-taxonomic relation vocabulary observed across the whole dataset
# (train + test splits; for the final submission all of it is training data).
# Fixed/hardcoded on purpose: deterministic and reviewable, and never derived from
# the challenge's hidden test set. Ordered most-frequent first; opaque RO ids last.
S5_RELATION_VOCAB = [
    "is defined by", "disjoint with", "equivalent class", "type", "exact match",
    "tree view", "part_of", "see also", "has part", "database_cross_reference",
    "same as", "is_conjugate_base_of", "is_conjugate_acid_of", "range",
    "derives from", "develops_from", "domain", "has role", "broader", "overlaps",
    "positively regulates", "regulates", "located in", "term replaced by",
    "ro_0002220", "ro_0002473",
]


# Richer alternative to the bare vocab list: one worked example per relation (from
# training) + usage conditions. The AXIOM relations (disjoint with / equivalent class)
# carry an explicit precision-guard because they account for ~56%/over-extraction —
# the model otherwise asserts disjointness between arbitrary type pairs by default.
S5_RELATION_GUIDE = """\
Valid relation labels — use one of these exact labels (a short verb phrase only if none \
fits). Each shows its meaning and a real example; assert a relation ONLY when clearly \
warranted by THIS document.

Stated/structural (assert when the passage states or strongly implies it):
- has part / part_of — composition. e.g. [agricultural experimental multiplot, has part, agricultural experimental plot]
- database_cross_reference — common name ↔ formal/scientific id or cross-ontology counterpart. e.g. [ambarella plant, database_cross_reference, spondias dulcis]
- exact match — identical concept across sources; usually a self-loop. e.g. [act of artifact processing, exact match, act of artifact processing]
- see also — related but not equivalent. e.g. [cowpea pulse food product, see also, cowpea vegetable food product]
- tree view — hierarchical structural nesting. e.g. [area of scrub, tree view, scrubland area]
- located in — spatial containment. e.g. [snow mass, located in, mountain]
- derives from / develops_from — origin/development. e.g. [bean substance, derives from, bean (whole)]
- term replaced by — obsolete term superseded by another.

Scaffolding (ontology serialization):
- is defined by — a class formally defined in this ontology; SELF-loop [X, is defined by, X]. e.g. [APIReference, is defined by, APIReference]
- type — a class declared as an individual: [X, type, "named individual"].

Axiom relations — NOT usually written in text; assert ONLY when the passage explicitly \
contrasts or equates the two classes, never by default:
- disjoint with — mutually exclusive classes (symmetric). e.g. [payment charge specification, disjoint with, unit price specification]
- equivalent class — synonymous/equivalent classes (symmetric). e.g. [absorption coefficient, equivalent class, opacity]
- same as — e.g. [country, same as, country]; broader — e.g. [sensortype, broader, tool]

Other valid labels (rare): is_conjugate_base_of, is_conjugate_acid_of, range, domain, \
has role, overlaps, positively regulates, regulates."""


def build_s5_messages(
    passage: str,
    typed_terms: list[tuple[str, str]],
    domain_description: str | None = None,
    examples: list[dict[str, Any]] | None = None,
    doc_types: list[str] | None = None,
    relation_vocab: list[str] | None = None,
    relation_guide: bool = False,
    axiom_guard: bool = False,
    prompt_v2: bool = False,
) -> list[dict[str, str]]:
    """Build messages for S5: combined text extraction + semantic type-pair inference.

    Args:
        passage: Source document text.
        typed_terms: List of ``(term, type)`` pairs known for this document.
        domain_description: Optional domain description from the domain
            inference pre-step. Prepended to the system prompt when provided.
        examples: Optional few-shot examples.
        doc_types: Optional explicit type vocabulary for the document. When
            provided it overrides the types derived from ``typed_terms`` (used to
            feed the full S2 type vocabulary, since S5 relations hold between types
            that may not all have a typed term).
        relation_vocab: Complete list of valid relation labels (the training
            vocabulary). Defaults to the fixed ``S5_RELATION_VOCAB``; pass ``[]`` to
            omit it. Appended to the system prompt as a closed-ish label set, with a
            free-phrase escape for unseen cases.
    """
    system = _S5_EXTRACT_SYSTEM_V2 if prompt_v2 else _S5_EXTRACT_SYSTEM
    if domain_description:
        system = f"Domain context: {domain_description}\n\n{system}"
    if prompt_v2:
        # V2 carries its own label list inline; skip the vocab/guide/axiom-guard appends.
        pass
    elif relation_guide:
        # Worked example + usage condition per relation (supersedes the bare list).
        system = system + "\n\n" + S5_RELATION_GUIDE
    else:
        vocab = relation_vocab if relation_vocab is not None else S5_RELATION_VOCAB
        if vocab:
            system = system + (
                "\n\nComplete list of valid relation labels — prefer one of these exact "
                "labels; only if none truly fits, use a short verb phrase from the text:\n"
                + ", ".join(vocab)
            )
    if axiom_guard and not prompt_v2:
        # Condition-only precision guard for the over-inferred axiom relations (no
        # examples, to avoid the anchoring that the full guide caused).
        system = system + (
            "\n\nBe especially conservative with the axiom relations \"disjoint with\" and "
            "\"equivalent class\": assert them ONLY when the passage explicitly states the "
            "two classes are mutually exclusive (disjoint) or the same/equivalent — never "
            "infer them by default from the type list."
        )

    # Unique types for this document — used for semantic inference
    if doc_types is None:
        doc_types = sorted({typ for _, typ in typed_terms}) if typed_terms else []
    doc_types_str = ", ".join(doc_types) if doc_types else "(none)"

    if typed_terms:
        typed_terms_str = "\n".join(f"- {term} ({typ})" for term, typ in typed_terms)
    else:
        typed_terms_str = "(none identified)"

    user_content = _S5_EXTRACT_USER.format(
        doc_types=doc_types_str,
        typed_terms=typed_terms_str,
        passage=passage.strip(),
    )

    if examples:
        def _triple(t: dict) -> str:
            return (f'  {{"subject": "{t["subject"]}", "relation": "{t["relation"]}", '
                    f'"object": "{t["object"]}"}}')

        if "triples" in examples[0]:
            # Grouped per-doc demonstrations: full types → relations mapping.
            blocks = []
            for i, e in enumerate(examples, 1):
                lines = [f"Example {i}:"]
                if e.get("text"):
                    lines.append(f"Document: {e['text'].strip()[:800]}")
                lines.append("Types: " + (", ".join(e.get("types", [])) or "(none)"))
                if e.get("triples"):
                    lines.append("Relations:")
                    lines += [_triple(t) for t in e["triples"]]
                else:
                    lines.append("Relations: (none)")
                blocks.append("\n".join(lines))
            ex_block = "\n\n".join(blocks)
        else:
            # Flat list of triples (legacy).
            ex_block = "\n".join(_triple(e) for e in examples[:50])

        user_content = (
            "Examples from similar documents — each shows a type vocabulary and the "
            "non-taxonomic relations that hold (note self-referential axioms like "
            '"is defined by" / "exact match", and "type" → "named individual"):\n\n'
            f"{ex_block}\n\n{user_content}"
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
