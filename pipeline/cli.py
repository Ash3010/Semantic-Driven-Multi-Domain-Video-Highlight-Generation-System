import argparse
from pathlib import Path
from pipeline.pipeline import run_highlights

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--mode", default="generic", choices=["generic", "sports", "lecture"])
    ap.add_argument("--target_seconds", type=int, default=75)
    ap.add_argument("--whisper_model", default="tiny", choices=["tiny","base","small"])
    args = ap.parse_args()

    out_dir = Path("outputs") / Path(args.video).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    res = run_highlights(
        video_path=args.video,
        youtube_url=None,
        out_dir=str(out_dir),
        mode=args.mode,
        target_seconds=args.target_seconds,
        whisper_model=args.whisper_model,
    )
    print("DONE")
    print(res)

if __name__ == "__main__":
    main()
