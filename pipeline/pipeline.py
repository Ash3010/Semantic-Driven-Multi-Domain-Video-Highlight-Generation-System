import json
from pathlib import Path

from .ffmpeg_utils import extract_audio
from .youtube import download_youtube
from .scenes import detect_scenes
from .transcribe import transcribe, text_in_range
from .audio_features import audio_energy, energy_in_range
from .visual_features import get_midframe_rgb_mean
from .semantic import build_embedder, embed_texts, cue_score, novelty_score
from .assemble import export_highlights
from .summarize import make_summary

def _weights(mode: str) -> dict:
    if mode == "sports":
        return dict(audio=0.45, visual=0.30, semantic=0.25)
    if mode == "lecture":
        return dict(audio=0.15, visual=0.15, semantic=0.70)
    return dict(audio=0.30, visual=0.25, semantic=0.45)

def run_highlights(
    video_path: str | None,
    youtube_url: str | None,
    out_dir: str,
    mode: str = "generic",
    target_seconds: int = 75,
    whisper_model: str = "tiny",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if youtube_url and (not video_path):
        try:
            video_path = download_youtube(youtube_url, str(out_dir / "tmp"))
        except Exception as e:
            raise RuntimeError(
            "YouTube download failed (YouTube often blocks automated downloads). "
            "For this demo, please download the video manually and upload the MP4.\n\n"
            f"Details: {e}"
        )

    if not video_path:
        raise ValueError("No video provided.")
    video_path = str(Path(video_path).resolve())

    wav_path = str(out_dir / "audio.wav")
    extract_audio(video_path, wav_path, sr=16000)

    transcript = transcribe(wav_path, model_size=whisper_model)
    scenes = detect_scenes(video_path)

    times, rms = audio_energy(wav_path)
    emb_model = build_embedder()
    weights = _weights(mode)

    scene_texts = [text_in_range(transcript, s.start_s, s.end_s) for s in scenes]
    scene_emb = embed_texts(emb_model, scene_texts) if any(t.strip() for t in scene_texts) else None

    picks_scored = []
    prev_emb = None

    for i, sc in enumerate(scenes):
        start, end = float(sc.start_s), float(sc.end_s)
        dur = max(0.1, end - start)

        a = energy_in_range(times, rms, start, end)
        v = get_midframe_rgb_mean(video_path, start, end)

        text = scene_texts[i]
        if scene_emb is not None and len(scene_emb) > i:
            emb = scene_emb[i]
            sem_novel = novelty_score(emb, prev_emb)
            sem_cue = cue_score(text, mode)
            sem_len = min(1.0, len(text) / 240.0)
            s_sem = (0.45 * sem_novel) + (0.35 * sem_cue) + (0.20 * sem_len)
            prev_emb = emb
        else:
            s_sem = 0.15

        a_n = min(1.0, a * 10.0)
        v_n = min(1.0, max(0.0, v))

        score = (weights["audio"] * a_n) + (weights["visual"] * v_n) + (weights["semantic"] * s_sem)

        picks_scored.append({
            "start": start,
            "end": end,
            "duration": dur,
            "score": float(score),
            "debug": f"audio={a_n:.3f}  visual={v_n:.3f}  semantic={s_sem:.3f}  text_len={len(text)}"
        })

    picks_scored.sort(key=lambda x: x["score"], reverse=True)
    selected = []
    total = 0.0
    for p in picks_scored:
        if total >= target_seconds:
            break
        seg_len = min(p["duration"], 18.0)
        start, end = p["start"], min(p["end"], p["start"] + seg_len)
        if end - start < 1.0:
            continue
        selected.append({**p, "start": start, "end": end, "duration": end - start})
        total += (end - start)

    selected.sort(key=lambda x: x["start"])

    highlights_path = str(out_dir / "highlights.mp4")
    export_highlights(video_path, selected, highlights_path)

    picked_texts = [text_in_range(transcript, p["start"], p["end"]) for p in selected]
    bullets, paragraph = make_summary(emb_model, picked_texts)

    timestamps_json = str(out_dir / "timestamps.json")
    with open(timestamps_json, "w", encoding="utf-8") as f:
        json.dump({"mode": mode, "target_seconds": target_seconds, "picks": selected}, f, indent=2)

    return {
        "highlights_path": highlights_path,
        "timestamps_json": timestamps_json,
        "summary_bullets": bullets,
        "summary_paragraph": paragraph,
        "picks": selected,
    }
