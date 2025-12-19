import numpy as np
from sentence_transformers import SentenceTransformer

CUE_PHRASES_LECTURE = [
    "important", "in summary", "to summarize", "key idea", "remember", "note that",
    "definition", "theorem", "we conclude", "in conclusion"
]
CUE_PHRASES_SPORTS = [
    "goal", "scores", "what a", "incredible", "unbelievable", "save", "touchdown",
    "three pointer", "knockout", "finish", "winner"
]

def build_embedder(name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(name)

def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)

def cue_score(text: str, mode: str) -> float:
    t = (text or "").lower()
    cues = CUE_PHRASES_LECTURE if mode == "lecture" else (CUE_PHRASES_SPORTS if mode == "sports" else (CUE_PHRASES_LECTURE + CUE_PHRASES_SPORTS))
    hits = sum(1 for c in cues if c in t)
    return min(1.0, hits / 3.0)

def novelty_score(curr_emb: np.ndarray, prev_emb: np.ndarray | None) -> float:
    if prev_emb is None:
        return 0.5
    sim = float(np.dot(curr_emb, prev_emb))  # embeddings are normalized
    return float(max(0.0, 1.0 - sim))
