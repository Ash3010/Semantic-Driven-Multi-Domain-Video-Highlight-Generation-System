import numpy as np
from moviepy.editor import VideoFileClip

def get_midframe_rgb_mean(video_path: str, start: float, end: float) -> float:
    t = float((start + end) / 2.0)
    with VideoFileClip(video_path) as clip:
        t = max(0.0, min(t, max(0.0, (clip.duration or 0.0) - 0.05)))
        frame = clip.get_frame(t)
    return float(np.mean(frame) / 255.0)
