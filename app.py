import os
import json
import uuid
import tempfile
from pathlib import Path

import streamlit as st

from pipeline.pipeline import run_highlights


CSS = "\n<style>\n.block-container {padding-top: 2rem; padding-bottom: 3rem;}\ndiv[data-testid=\"stMetric\"] {background: #ffffff; border: 1px solid rgba(15,23,42,0.08); border-radius: 18px; padding: 12px;}\ndiv[data-testid=\"stExpander\"] {border-radius: 18px; border: 1px solid rgba(15,23,42,0.08); overflow: hidden;}\n.stButton>button {border-radius: 14px; padding: 0.65rem 1rem; border: 1px solid rgba(15,23,42,0.10);}\n.stDownloadButton>button {border-radius: 14px; padding: 0.65rem 1rem;}\nsmall.muted {color: rgba(15,23,42,0.65);}\nhr {border-color: rgba(15,23,42,0.08);}\n</style>\n"

st.set_page_config(
    page_title="Multi‑Domain Highlights & Semantic Summary",
    layout="wide",
)

st.markdown(CSS, unsafe_allow_html=True)

st.title("Multi‑Domain Video Highlights & Semantic Summary")
st.caption("Upload any video (sports / lectures / generic). This demo uses **semantic embeddings** + audio/visual cues to pick the best moments.")

colA, colB = st.columns([1.05, 0.95], gap="large")

with colA:
    st.subheader("1) Input")
    upl = st.file_uploader("Upload a video file", type=["mp4", "mov", "mkv", "webm"])
    yt = st.text_input("…or YouTube link", placeholder="https://www.youtube.com/watch?v=...")

    st.subheader("2) Settings")
    mode = st.selectbox("Mode", ["generic", "sports", "lecture"], index=0,
                        help="Changes scoring weights for different domains.")
    target_seconds = st.slider("Target highlights length (seconds)", 20, 180, 75, 5)
    whisper_model = st.selectbox("Whisper model (speed vs accuracy)", ["tiny", "base", "small"], index=0)
    add_debug = st.checkbox("Show debug scores", value=True)

    run_btn = st.button("🚀 Generate Highlights", use_container_width=True)

with colB:
    st.subheader("Result:")
    st.markdown(
        "- **highlights.mp4** (download)\n"
        "- **timestamps.json** (why each moment was chosen)\n"
        "- **semantic summary** (bullets + short paragraph)"
    )

def save_upload_to_temp(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return tmp_path

if run_btn:
    if not upl and not yt:
        st.error("Upload a video or paste a YouTube link.")
        st.stop()

    run_id = uuid.uuid4().hex[:8]
    out_dir = Path("outputs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = None
    if upl:
        video_path = save_upload_to_temp(upl)

    with st.status("Running pipeline…", expanded=True) as status:
        try:
            result = run_highlights(
                video_path=video_path,
                youtube_url=yt.strip() if yt else None,
                out_dir=str(out_dir),
                mode=mode,
                target_seconds=int(target_seconds),
                whisper_model=whisper_model,
            )
            status.update(label="Done ✅", state="complete")
        except Exception as e:
            status.update(label="Failed ❌", state="error")
            st.exception(e)
            st.stop()

    st.divider()
    st.subheader(" Results")

    if result.get("highlights_path") and Path(result["highlights_path"]).exists():
        st.video(result["highlights_path"])
        with open(result["highlights_path"], "rb") as f:
            st.download_button(" Download highlights.mp4", f, file_name="highlights.mp4", use_container_width=True)

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("###  Semantic summary")
        st.write(result.get("summary_paragraph", ""))
        bullets = result.get("summary_bullets", [])
        if bullets:
            st.markdown("\n".join([f"- {x}" for x in bullets]))
    with c2:
        st.markdown("###  Timeline")
        picks = result.get("picks", [])
        if picks:
            for p in picks:
                st.markdown(f"**{p['start']:.1f}s → {p['end']:.1f}s**  ·  score: `{p['score']:.3f}`")
                if add_debug and p.get("debug"):
                    st.caption(p["debug"])
        else:
            st.caption("No segments selected.")

    if result.get("timestamps_json") and Path(result["timestamps_json"]).exists():
        with open(result["timestamps_json"], "rb") as f:
            st.download_button(" Download timestamps.json", f, file_name="timestamps.json", use_container_width=True)

    st.success(f"Saved outputs to: {out_dir}")
