from dataclasses import dataclass
from typing import List
from faster_whisper import WhisperModel

@dataclass
class TranscriptSeg:
    start: float
    end: float
    text: str

def transcribe(wav_path: str, model_size: str = "tiny") -> List[TranscriptSeg]:
    model = WhisperModel(model_size, device="auto", compute_type="int8")
    segments, _info = model.transcribe(wav_path, vad_filter=True)
    out: List[TranscriptSeg] = []
    for s in segments:
        t = (s.text or "").strip()
        if t:
            out.append(TranscriptSeg(float(s.start), float(s.end), t))
    return out

def text_in_range(segs: List[TranscriptSeg], start: float, end: float) -> str:
    parts = []
    for s in segs:
        if s.end <= start:
            continue
        if s.start >= end:
            break
        parts.append(s.text)
    return " ".join(parts).strip()
