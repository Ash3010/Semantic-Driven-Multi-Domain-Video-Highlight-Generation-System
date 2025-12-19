import subprocess
from pathlib import Path

YTDLP = "yt-dlp"

def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n\n{p.stderr}")

def download_youtube(url: str, out_dir: str) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_tpl = str(Path(out_dir) / "youtube_input.%(ext)s")
    _run([YTDLP, "-f", "mp4/best", "-o", out_tpl, url])
    files = list(Path(out_dir).glob("youtube_input.*"))
    if not files:
        raise RuntimeError("yt-dlp finished but output file was not found.")
    return str(files[0])
