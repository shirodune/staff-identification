"""Gradio front-end for detect_staff.py (Hugging Face Space).
Upload a top-down sensor video -> annotated video, per-frame CSV, presence timeline."""
import csv, json, os, subprocess, sys, tempfile, time
from pathlib import Path
import gradio as gr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio_ffmpeg

HERE = Path(__file__).parent
MAX_SECONDS = 180

# Works on a CPU Space as is. On a ZeroGPU Space the main function must carry @spaces.GPU,
# otherwise the Space refuses to start ("No @spaces.GPU function detected").
try:
    import spaces
    gpu = spaces.GPU(duration=120)
except Exception:
    gpu = lambda f: f

@gpu
def run(video, fast, progress=gr.Progress()):
    if video is None: raise gr.Error("Please upload a video.")
    import cv2
    cap = cv2.VideoCapture(video); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 25; cap.release()
    if n / fps > MAX_SECONDS: raise gr.Error(f"Video is {n/fps:.0f}s; please keep it under {MAX_SECONDS}s (CPU Space).")
    out = Path(tempfile.mkdtemp())
    cmd = [sys.executable, str(HERE / "detect_staff.py"), "--video", video, "--out", str(out)]
    if fast: cmd += ["--person_imgsz", "640"]
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = []
    for line in proc.stdout:
        log.append(line.rstrip())
        if line.startswith("  frame"):
            f = int(line.split()[1]); progress(min(0.95, f / max(1, n)), desc=f"frame {f}/{n}  ({time.time()-t0:.0f}s)")
    proc.wait()
    if proc.returncode != 0: raise gr.Error("Pipeline failed:\n" + "\n".join(log[-15:]))
    # browser-playable video
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    h264 = out / "annotated_h264.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(out / "annotated.mp4"), "-c:v", "libx264", "-crf", "26", "-preset", "fast", "-pix_fmt", "yuv420p", str(h264)], check=True)
    # timeline
    rows = list(csv.DictReader(open(out / "staff_frames.csv")))
    present = np.array([int(r["staff_present"]) for r in rows]); seen = np.array([int(r["badge_visible"]) for r in rows])
    fig, ax = plt.subplots(figsize=(12, 1.8))
    ax.fill_between(np.arange(len(seen)), 0.55, 0.95, where=seen > 0, step="post", color="#e9c46a", lw=0)
    ax.fill_between(np.arange(len(present)), 0.05, 0.45, where=present > 0, step="post", color="#e76f51", lw=0)
    ax.text(-5, 0.75, "badge seen", ha="right", va="center", fontsize=9); ax.text(-5, 0.25, "staff present", ha="right", va="center", fontsize=9)
    ax.set_xlim(0, len(present)); ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xlabel("frame")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    plt.tight_layout(); fig.savefig(out / "timeline.png", dpi=120); plt.close(fig)
    summ = json.load(open(out / "summary.json"))
    segs, s = [], None
    for i, v in enumerate(list(present) + [0]):
        if v and s is None: s = i
        if not v and s is not None: segs.append(f"{s/fps:.1f}s–{(i-1)/fps:.1f}s"); s = None
    text = (f"{len(summ['staff'])} staff member(s) found · present in {int(present.sum())}/{len(present)} frames "
            f"({present.mean()*100:.0f} %) · badge seen in {int(seen.sum())} frames · {summ['runtime_s']} s\n"
            f"present: {', '.join(segs) if segs else '—'}")
    return str(h264), str(out / "staff_frames.csv"), str(out / "timeline.png"), text

with gr.Blocks(title="Staff identification") as demo:
    gr.Markdown("""# Staff identification from a top-down sensor video
Tracks people (Ultralytics YOLO26 + ByteTrack), looks for the staff **name badge** inside each person box with a small
custom detector, then picks the badge-wearer's trajectory as the highest-scoring path through the person boxes.
Output: annotated video, per-frame `staff_present / x / y` CSV, presence timeline. CPU only — about 8 min per minute of
video at full resolution, ~4 min in fast mode.""")
    with gr.Row():
        with gr.Column():
            vid = gr.Video(label="Input video (top-down sensor, ≤ 3 min)")
            fast = gr.Checkbox(label="Fast mode (640 px person detector, ~2× faster, more misses)", value=False)
            btn = gr.Button("Run", variant="primary")
            if (HERE / "data" / "sample.mp4").exists():   # optional demo clip
                gr.Examples([[str(HERE / "data" / "sample.mp4"), True]], inputs=[vid, fast])
        with gr.Column():
            out_vid = gr.Video(label="Annotated video (red = staff, yellow = badge seen)")
            out_txt = gr.Textbox(label="Summary", lines=3)
            out_tl = gr.Image(label="Timeline")
            out_csv = gr.File(label="staff_frames.csv")
    btn.click(run, [vid, fast], [out_vid, out_csv, out_tl, out_txt])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
