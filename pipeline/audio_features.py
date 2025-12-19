import numpy as np
import soundfile as sf

def audio_energy(wav_path: str, frame_ms: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      times: time (seconds) for each frame
      rms: RMS energy per frame
    """
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)  # mono

    frame_len = int(sr * (frame_ms / 1000.0))
    frame_len = max(frame_len, 1)

    n_frames = int(np.ceil(len(y) / frame_len))
    padded = np.pad(y, (0, n_frames * frame_len - len(y)), mode="constant")

    frames = padded.reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

    times = (np.arange(n_frames) * frame_len) / sr
    return times, rms

def energy_in_range(times: np.ndarray, rms: np.ndarray, start: float, end: float) -> float:
    if len(times) == 0:
        return 0.0
    mask = (times >= start) & (times <= end)
    if not mask.any():
        return 0.0
    return float(np.mean(rms[mask]))
