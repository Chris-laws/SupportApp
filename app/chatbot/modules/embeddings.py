from __future__ import annotations

import os
import pickle
from typing import Iterable, List, Sequence, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _get_model(name: str) -> SentenceTransformer:
    model = _MODEL_CACHE.get(name)
    if model is None:
        model = SentenceTransformer(name)
        _MODEL_CACHE[name] = model
    return model


def get_current_embedding_model() -> str:
    return os.getenv("SUPPORTAPP_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")



def get_embeddings(texts: Sequence[str], normalize: bool = True) -> np.ndarray:
    model_name = get_current_embedding_model()
    model = _get_model(model_name)
    embeddings = model.encode(list(texts), show_progress_bar=False)
    if normalize:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings.astype(np.float32)



def save_embeddings_to_faiss(embeddings: np.ndarray, chunk_records: Iterable[dict], save_path_base: str) -> None:
    os.makedirs(os.path.dirname(save_path_base), exist_ok=True)
    embeddings = embeddings.astype(np.float32)
    dimension = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = 128
    index.add(embeddings)
    faiss.write_index(index, save_path_base + ".index")

    with open(save_path_base + ".pkl", "wb") as f:
        pickle.dump(list(chunk_records), f)

    np.save(save_path_base + ".npy", embeddings)



def load_faiss_index(load_path_base: str) -> Tuple[faiss.Index | None, List[dict], np.ndarray | None]:
    print("Lade FAISS-Index...")
    chunk_records: List[dict] = []
    embeddings: np.ndarray | None = None
    try:
        index = faiss.read_index(load_path_base + ".index")
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = 128
        with open(load_path_base + ".pkl", "rb") as f:
            chunk_records = pickle.load(f)
        embeddings_path = load_path_base + ".npy"
        if os.path.exists(embeddings_path):
            embeddings = np.load(embeddings_path)
        return index, chunk_records, embeddings
    except Exception as exc:  # noqa: BLE001
        print(f"Fehler beim Laden des FAISS-Index: {exc}")
        return None, chunk_records, embeddings



def get_most_similar_chunks(question_embedding: np.ndarray, index: faiss.Index, chunk_records: Sequence[dict], top_k: int = 15) -> List[dict]:
    if index is None or not chunk_records:
        return []

    query = np.array([question_embedding], dtype=np.float32)
    distances, indices = index.search(query, top_k)

    similar_chunks: List[dict] = []
    for score, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(chunk_records):
            record = dict(chunk_records[idx])
            record.update({
                "score": float(score),
                "index": int(idx),
            })
            similar_chunks.append(record)
    return similar_chunks
