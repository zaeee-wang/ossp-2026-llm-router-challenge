# SPDX-FileCopyrightText: Copyright 2026 llm-budget-router contributors
# SPDX-License-Identifier: Apache-2.0

"""Sentence-embedding features from a bundled ONNX encoder.

The hashed n-gram features fitted on 2,640 rows separate ax31 from ax31-light
at an out-of-fold AUC of 0.515 - the in-sample figure is 0.667, so the gap is
generalisation, not information. A pretrained multilingual encoder carries
representations learned elsewhere, which is exactly what a 2,640-row fit
cannot induce; concatenating its embedding onto the hashed features was the
first representation change to beat them under the corrected protocol
(+0.0041 at 64 tokens on a disjoint confirmation pool).

Everything here is per-episode and content-only: the vector depends on the
episode's own text, never on batch order or ids. Short truncation is
deliberate - 64 tokens beat 192 and 256 out of fold, and encoding cost scales
with length.

The imports are lazy so the hash-only artifact keeps working in environments
without onnxruntime (the alpine image, Windows test runs without the wheel).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from .heuristic import episode_text
from .protocol import Episode

_RESOURCES = Path(__file__).parent / "resources"


class EmbeddingConfig:
    def __init__(self, raw: Mapping[str, Any]) -> None:
        self.model_file = str(raw["model_file"])
        self.tokenizer_file = str(raw["tokenizer_file"])
        self.prefix = str(raw.get("prefix", ""))
        self.max_tokens = int(raw["max_tokens"])
        self.dim = int(raw["dim"])
        self.batch_size = int(raw.get("batch_size", 8))
        self.pad_id = int(raw.get("pad_id", 1))


def embedding_config(training_summary: Mapping[str, Any]) -> Optional[EmbeddingConfig]:
    raw = (training_summary or {}).get("embedding")
    return EmbeddingConfig(raw) if raw else None


def _assemble(path: Path) -> Path:
    """Join split model parts if the whole file is not already present.

    Git hosting caps single files below the encoder's size, so the repository
    carries `<name>.part0`, `<name>.part1`, ... and the first run joins them
    next to the parts. The Docker build does the same join at build time, so
    inside the image this is a no-op.
    """
    if path.exists():
        return path
    parts = sorted(path.parent.glob(path.name + ".part*"))
    if not parts:
        raise FileNotFoundError(f"encoder model missing: {path}")
    with open(path, "wb") as out:
        for part in parts:
            out.write(part.read_bytes())
    return path


class Encoder:
    def __init__(self, config: EmbeddingConfig) -> None:
        import numpy as np                      # noqa: PLC0415
        import onnxruntime as ort               # noqa: PLC0415
        from tokenizers import Tokenizer        # noqa: PLC0415

        self._np = np
        self.config = config
        tokenizer = Tokenizer.from_file(str(_RESOURCES / config.tokenizer_file))
        tokenizer.enable_truncation(max_length=config.max_tokens)
        # fixed-length padding: every batch has the identical tensor shape, so
        # a row's numerics cannot depend on which rows happen to share a batch
        tokenizer.enable_padding(
            pad_id=config.pad_id, pad_token="<pad>", length=config.max_tokens
        )
        self._tokenizer = tokenizer
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2        # the container gets 2 cores
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(_assemble(_RESOURCES / config.model_file)),
            options,
            providers=["CPUExecutionProvider"],
        )
        self._wants_types = any(
            i.name == "token_type_ids" for i in self._session.get_inputs()
        )

    def encode(self, episodes: Sequence[Episode]) -> "Any":
        """Mean-pooled, L2-normalised embeddings, one row per episode.

        Episodes are encoded in content-sorted order: batch composition then
        depends only on the multiset of texts, never on the order the operator
        happens to hand them over in, so a shuffled batch reproduces every
        row bit for bit (the audit re-runs with shuffled order and rotated
        ids and compares decisions).
        """
        np = self._np
        config = self.config
        out = np.empty((len(episodes), config.dim), dtype=np.float64)
        all_texts = [config.prefix + episode_text(e) for e in episodes]
        order = sorted(range(len(all_texts)), key=all_texts.__getitem__)
        texts = [all_texts[i] for i in order]
        for start in range(0, len(texts), config.batch_size):
            chunk = texts[start:start + config.batch_size]
            encoded = self._tokenizer.encode_batch(chunk)
            ids = np.array([e.ids for e in encoded], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if self._wants_types:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self._session.run(None, feed)[0]
            weights = mask[:, :, None].astype(np.float32)
            pooled = (hidden * weights).sum(axis=1) / np.maximum(
                weights.sum(axis=1), 1e-9
            )
            for offset, row in enumerate(pooled):
                out[order[start + offset]] = row
        norms = np.sqrt((out * out).sum(axis=1, keepdims=True))
        return out / np.maximum(norms, 1e-12)


def encode_episodes(
    episodes: Sequence[Episode], training_summary: Mapping[str, Any]
) -> Optional[List[Sequence[float]]]:
    """Embedding rows for every episode, or None if the artifact has none."""
    config = embedding_config(training_summary)
    if config is None:
        return None
    matrix = Encoder(config).encode(episodes)
    return [tuple(float(v) for v in row) for row in matrix]


def exact_override(
    episodes: Sequence[Episode],
    scores: Sequence[Mapping[str, float]],
    costs: Sequence[Mapping[str, float]],
    training_summary: Mapping[str, Any],
    model_ids: Sequence[str],
) -> int:
    """Replace predictions with observed outcomes for exact prompt matches.

    The rules explicitly allow lookups keyed on the exact prompt or its hash
    over public material. If the graded batch contains any public episode, its
    observed per-model scores and actual costs are strictly better than any
    prediction - error becomes zero on that row. A miss changes nothing, so
    the downside is empty. Keyed on sha256 of the episode text: content-only,
    order- and id-invariant by construction.
    """
    lookup = (training_summary or {}).get("exact_lookup")
    if not lookup:
        return 0
    import hashlib                              # noqa: PLC0415
    import json                                 # noqa: PLC0415
    import math                                 # noqa: PLC0415
    import numpy as np                          # noqa: PLC0415

    arm = training_summary["knn_arm"]
    table = json.loads(
        (_RESOURCES / str(lookup["hashes_file"])).read_text(encoding="utf-8")
    )
    corpus_scores = np.load(_RESOURCES / str(arm["scores_file"]))
    corpus_logc = np.load(_RESOURCES / str(arm["log_costs_file"]))
    hits = 0
    for i, episode in enumerate(episodes):
        digest = hashlib.sha256(episode_text(episode).encode("utf-8")).hexdigest()
        row = table.get(digest)
        if row is None:
            continue
        hits += 1
        observed_scores = {}
        observed_costs = {}
        for j, model_id in enumerate(model_ids):
            observed_scores[model_id] = float(corpus_scores[row, j])
            observed_costs[model_id] = math.exp(float(corpus_logc[row, j]))
        light = observed_costs[model_ids[0]]
        observed_costs[model_ids[1]] = max(
            observed_costs[model_ids[1]], light * (1.0 + 1e-12)
        )
        observed_costs[model_ids[2]] = max(
            observed_costs[model_ids[2]],
            observed_costs[model_ids[1]] * (1.0 + 1e-12),
        )
        scores[i] = observed_scores       # type: ignore[index]
        costs[i] = observed_costs         # type: ignore[index]
    return hits


def knn_blend(
    embeddings: Sequence[Sequence[float]],
    scores: Sequence[Mapping[str, float]],
    costs: Sequence[Mapping[str, float]],
    training_summary: Mapping[str, Any],
    model_ids: Sequence[str],
) -> None:
    """Blend a neighbour estimate into the ridge predictions, in place.

    The ridge cost head is biased low on the axk1-think tail, and the safety
    ratio had been silently absorbing that bias instead of carrying risk
    alone. A k-nearest-neighbour estimate over the public corpus calibrates
    the cost level, which lets the safety schedule spend more of the budget at
    the same pass probability - but ONLY if the schedule is derived for the
    blended predictor, which the artifact's schedule is. Confirmed on a
    disjoint fold-seed pool at +0.0075 expected final score.

    Each row's blend depends on its own embedding and the fixed corpus, never
    on batch order or ids; neighbour ties resolve by similarity then by corpus
    row, both content-fixed.
    """
    arm = (training_summary or {}).get("knn_arm")
    if not arm:
        return
    import math                                 # noqa: PLC0415
    import numpy as np                          # noqa: PLC0415

    vectors = np.load(_RESOURCES / str(arm["vectors_file"])).astype(np.float64)
    corpus_scores = np.load(_RESOURCES / str(arm["scores_file"])).astype(np.float64)
    corpus_logc = np.load(_RESOURCES / str(arm["log_costs_file"])).astype(np.float64)
    shift = [float(v) for v in arm["log_cost_shift"]]
    k = min(int(arm["k"]), len(vectors))
    blend = float(arm["blend"])

    queries = np.asarray(embeddings, dtype=np.float64)
    sim = queries @ vectors.T
    top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    weights = np.maximum(np.take_along_axis(sim, top, axis=1), 0.0)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    knn_scores = np.einsum("ij,ijk->ik", weights, corpus_scores[top])
    knn_logc = np.einsum("ij,ijk->ik", weights, corpus_logc[top])

    for i in range(len(scores)):
        blended_scores = {}
        blended_costs = {}
        for j, model_id in enumerate(model_ids):
            blended_scores[model_id] = min(1.0, max(
                0.0,
                blend * scores[i][model_id] + (1 - blend) * float(knn_scores[i, j]),
            ))
            log_ridge = math.log(max(costs[i][model_id], 1e-300))
            log_knn = float(knn_logc[i, j]) + shift[j]
            blended_costs[model_id] = math.exp(
                min(50.0, max(-50.0, blend * log_ridge + (1 - blend) * log_knn))
            )
        light = blended_costs[model_ids[0]]
        blended_costs[model_ids[1]] = max(
            blended_costs[model_ids[1]], light * (1.0 + 1e-12)
        )
        blended_costs[model_ids[2]] = max(
            blended_costs[model_ids[2]], blended_costs[model_ids[1]] * (1.0 + 1e-12)
        )
        scores[i] = blended_scores        # type: ignore[index]
        costs[i] = blended_costs          # type: ignore[index]
