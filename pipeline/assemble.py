from pathlib import Path
from moviepy.editor import VideoFileClip, concatenate_videoclips

def export_highlights(video_path: str, picks: list[dict], out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    clips = []
    with VideoFileClip(video_path) as full:
        for p in picks:
            s, e = float(p["start"]), float(p["end"])
            s = max(0.0, min(s, full.duration))
            e = max(0.0, min(e, full.duration))
            if e - s <= 0.2:
                continue
            clips.append(full.subclip(s, e))
        if not clips:
            clips = [full.subclip(0, min(10, full.duration))]
        out = concatenate_videoclips(clips, method="compose")
        out.write_videofile(out_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
    return out_path
