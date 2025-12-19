import numpy as np
from .semantic import embed_texts

def make_summary(emb_model, picked_texts: list[str], max_bullets: int = 6) -> tuple[list[str], str]:
    picked_texts = [t.strip() for t in picked_texts if t and t.strip()]
    if not picked_texts:
        return [], "No speech transcript detected in the selected highlights. Highlights were chosen primarily from audio/visual cues."

    emb = embed_texts(emb_model, picked_texts)
    centroid = emb.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

    sims = [(float(np.dot(emb[i], centroid)), i) for i in range(len(picked_texts))]
    sims.sort(reverse=True)

    bullets = []
    for _sim, idx in sims[:max_bullets * 2]:
        txt = picked_texts[idx].replace("  ", " ").strip()
        if len(txt) > 180:
            txt = txt[:177].rsplit(" ", 1)[0] + "…"
        if txt not in bullets:
            bullets.append(txt)
        if len(bullets) >= max_bullets:
            break

    paragraph = " ".join([b.replace("…", "") for b in bullets[:3]])
    return bullets, paragraph
