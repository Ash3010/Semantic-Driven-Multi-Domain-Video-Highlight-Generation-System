from dataclasses import dataclass
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

@dataclass
class Scene:
    start_s: float
    end_s: float

def _to_seconds(t) -> float:
    """Handles FrameTimecode-like objects, tuples, and raw numbers."""
    if hasattr(t, "get_seconds"):
        return float(t.get_seconds())
    if isinstance(t, tuple) and len(t) >= 1:
        # Some versions return (seconds, frames) or similar
        return float(t[0])
    try:
        return float(t)
    except Exception:
        return 0.0

def detect_scenes(video_path: str, threshold: float = 27.0) -> list[Scene]:
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    scenes: list[Scene] = []
    for start, end in scene_list:
        scenes.append(Scene(_to_seconds(start), _to_seconds(end)))

    # Robust duration (avoid get_duration().get_seconds() incompatibilities)
    duration = _to_seconds(video_manager.get_duration())
    video_manager.release()

    if not scenes:
        scenes = [Scene(0.0, float(duration) if duration > 0 else 10.0)]
    return scenes
