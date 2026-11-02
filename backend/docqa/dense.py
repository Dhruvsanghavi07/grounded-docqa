"""Dense retrieval.

Default: LSA (TF-IDF followed by truncated SVD), which runs anywhere with
scikit-learn and needs no model download. Optional: a sentence-transformers
model, enabled with DOCQA_EMBEDDER=st:<model-name>. Both return L2-normalized
vectors so cosine similarity is a dot product.
"""

from __future__ import annotations

import numpy as np


class LSAEmbedder:
    name = "lsa"

    def __init__(self, dims: int = 256):
        self.dims = dims
        self._tfidf = None
        self._svd = None

    def fit(self, texts: list[str]) -> LSAEmbedder:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1, stop_words="english")
        x = self._tfidf.fit_transform(texts)
        dims = max(2, min(self.dims, x.shape[1] - 1, x.shape[0] - 1))
        self._svd = TruncatedSVD(n_components=dims, random_state=0).fit(x)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self._svd.transform(self._tfidf.transform(texts))
        return _normalize(v)


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.name = f"st:{model}"
        self._model = SentenceTransformer(model)

    def fit(self, texts: list[str]) -> SentenceTransformerEmbedder:
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.asarray(self._model.encode(texts, batch_size=64, show_progress_bar=False)))


def make_embedder(spec: str = "lsa"):
    if spec.startswith("st:"):
        return SentenceTransformerEmbedder(spec[3:])
    return LSAEmbedder()


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n == 0, 1, n)
