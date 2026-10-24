from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from vision_engine import VisionEngine


# app.py, vision_engine.py, yolov8n.pt, and the models/ folder are expected
# to live under the same project directory.
BASE = Path(__file__).resolve().parent


class VisionVideoProcessor(VideoProcessorBase):
    """Run the existing VisionEngine on each browser camera frame."""

    def __init__(self, engine):
        self.engine = engine

    def recv(self, frame):
        import av

        image = frame.to_ndarray(format="bgr24")
        image = self.engine.process_frame(image)
        return av.VideoFrame.from_ndarray(image, format="bgr24")


st.set_page_config(
    page_title="VisionAI | Intelligent Vision",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: radial-gradient(ellipse at top left, #17213a 0%, #0b1020 44%, #080b14 100%); }
    [data-testid="stSidebar"] { background: #0c1221; border-right: 1px solid #27334a; }
    .block-container { padding-top: 1.5rem; max-width: 1600px; }
    .hero { padding: 1.25rem 1.5rem; border: 1px solid #26344c; border-radius: 18px; background: linear-gradient(115deg, rgba(99,91,255,.20), rgba(18,28,48,.85)); margin-bottom: 1rem; }
    .hero h1 { margin: 0; letter-spacing: -1px; font-size: 2.15rem; }
    .hero p { color: #aab6ce; margin: .35rem 0 0 0; }
    .metric-card { background: rgba(19,28,47,.9); border: 1px solid #283750; padding: 1rem 1.1rem; border-radius: 15px; min-height: 115px; }
    .metric-label { color: #9eacc4; font-size: .82rem; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #f4f7ff; margin-top: .3rem; }
    .metric-sub { color: #8b9bb6; font-size: .75rem; }
    .section-title { font-size: 1.1rem; font-weight: 650; margin: .4rem 0 .8rem; }
    div[data-testid="stButton"] button { border-radius: 10px; min-height: 2.5rem; }
    div[data-testid="stFileUploader"] section { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine():
    return VisionEngine(BASE)


def metric(label, value, sub):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def read_history():
    path = BASE / "unknown_log.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Timestamp", "Name", "Snapshot", "Similarity"])
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame(columns=["Timestamp", "Name", "Snapshot", "Similarity"])


try:
    engine = get_engine()
except Exception as exc:
    st.error(f"Could not initialize VisionAI: {exc}")
    st.info(
        "Check that yolov8n.pt is beside app.py and both ONNX model files "
        "are inside the models/ folder."
    )
    st.stop()


@st.fragment(run_every=0.5)
def render_live_status():
    current = engine.snapshot()

    if current.get("last_error"):
        st.error(current["last_error"])

    x, y, z, q = st.columns(4)
    x.metric("Objects", current.get("object_count", 0))
    y.metric("Faces", current.get("face_count", 0))
    z.metric("Known faces", current.get("known_count", 0))
    q.metric("Unknown faces", current.get("unknown_count", 0))

    st.caption(
        f"FPS: {current.get('fps', 0):.1f} | "
        f"YOLO: {current.get('yolo_ms', 0):.1f} ms | "
        f"Face recognition: {current.get('face_ms', 0):.1f} ms"
    )


st.sidebar.markdown("## 👁️ VisionAI")
st.sidebar.caption("INTELLIGENT VISION SYSTEM")
page = st.sidebar.radio(
    "WORKSPACE",
    ["Overview", "Live Monitoring", "Face Management", "Detection History"],
    label_visibility="visible",
)
st.sidebar.divider()
st.sidebar.caption("YOLOv8 · YuNet · SFace")
st.sidebar.caption("Local processing · CPU/GPU auto-select")

st.markdown(
    '<div class="hero"><h1>👁️ VisionAI</h1>'
    '<p>Real-time object detection and face recognition · Local AI workspace</p></div>',
    unsafe_allow_html=True,
)

if page == "Overview":
    state = engine.snapshot()
    a, b, c, d = st.columns(4)
    with a:
        metric("OBJECTS IN CURRENT FRAME", state["object_count"], "YOLOv8 detections")
    with b:
        metric("FACES IN CURRENT FRAME", state["face_count"], "YuNet face detector")
    with c:
        metric(
            "REGISTERED FACE IMAGES",
            state["registered_images"],
            f'{len(state["people"])} registered people',
        )
    with d:
        metric("UNKNOWN FACES IN FRAME", state["unknown_count"], "Unknown identity matches")

    left, right = st.columns([1.6, 1])
    with left:
        st.markdown('<div class="section-title">System status</div>', unsafe_allow_html=True)
        if state["running"]:
            st.success(f"Camera is running · inference device: {engine.device.upper()}")
        else:
            st.warning("Camera is stopped. Open Live Monitoring to start it.")
        if state["last_error"]:
            st.error(state["last_error"])
        st.markdown(
            "**Model pipeline:** YOLOv8n → object labels · YuNet → face detection · "
            "SFace → face matching"
        )
        st.caption("Counts describe the latest processed frame, not lifetime totals.")

    with right:
        st.markdown('<div class="section-title">Recent unknown-face events</div>', unsafe_allow_html=True)
        hist = read_history()
        if not hist.empty:
            st.dataframe(hist.tail(5).iloc[::-1], use_container_width=True, hide_index=True)
        else:
            st.info("No unknown-face events logged yet.")

elif page == "Live Monitoring":
    st.markdown('<div class="section-title">Live camera monitor</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    confidence = col1.slider("Object confidence", 0.10, 0.95, 0.35, 0.05)
    threshold = col2.slider(
        "Face similarity threshold",
        0.20,
        0.80,
        0.363,
        0.01,
        help="Higher values are stricter. Validate with your own test images.",
    )

    # Apply UI controls to the engine; camera selection is handled by the browser/WebRTC.
    engine.update_settings(
        confidence=float(confidence),
        similarity_threshold=float(threshold),
    )

    status_col = st.container()
    webrtc_ctx = webrtc_streamer(
        key="visionai-camera",
        video_processor_factory=lambda: VisionVideoProcessor(engine),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.state.playing:
        status_col.success("LIVE · camera stream active")
    else:
        status_col.info("Click START in the video panel to enable the camera.")

    render_live_status()
    st.caption("Live video is processed by WebRTC; metrics refresh independently.")

elif page == "Face Management":
    st.markdown('<div class="section-title">Register and manage identities</div>', unsafe_allow_html=True)
    st.info(
        "Only add reference photos for people who have agreed to participate. "
        "Images are stored locally in known_faces/."
    )
    with st.form("register_form", clear_on_submit=True):
        name = st.text_input("Person name / display label", placeholder="e.g. Alex")
        uploads = st.file_uploader(
            "Upload reference photos",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("＋ Register face images", use_container_width=True)
        if submitted:
            if not name.strip() or not uploads:
                st.warning("Enter a name and upload at least one image.")
            else:
                try:
                    safe_name, saved_count = engine.register_uploaded_images(name, uploads)
                    valid_features = engine.snapshot()["registered_images"]
                    st.success(
                        f"Saved {saved_count} image(s) for {safe_name}. "
                        f"Recognizable face features loaded across the database: {valid_features}."
                    )
                except Exception as exc:
                    st.error(str(exc))

    st.divider()
    people = engine.people()
    st.subheader(f"Registered people ({len(people)})")
    if not people:
        st.info("No people registered yet. Upload reference photos above.")

    for person in people:
        folder = BASE / "known_faces" / person
        if not folder.exists():
            continue
        files = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in VisionEngine.IMAGE_EXTENSIONS
        ]
        with st.container(border=True):
            c1, c2, c3 = st.columns([2, 2, 1])
            c1.markdown(f"**{person}**")
            c2.caption(f"{len(files)} stored image(s)")
            if c3.button("Delete", key=f"delete_{person}"):
                engine.delete_person(person)
                st.rerun()
            with st.expander(f"View photos · {person}"):
                if not files:
                    st.caption("No image files found.")
                else:
                    cols = st.columns(min(4, len(files)))
                    for i, file in enumerate(files):
                        try:
                            cols[i % len(cols)].image(
                                str(file), caption=file.name, use_container_width=True
                            )
                        except Exception:
                            continue

elif page == "Detection History":
    st.markdown('<div class="section-title">Unknown-face event history</div>', unsafe_allow_html=True)
    history = read_history()
    if history.empty:
        st.info("No events have been logged yet. Unknown-face logging starts during live monitoring.")
    else:
        c1, c2 = st.columns([3, 1])
        query = c1.text_input("Filter by name or timestamp")
        if query:
            mask = history.astype(str).apply(
                lambda col: col.str.contains(query, case=False, na=False, regex=False)
            ).any(axis=1)
            history = history[mask]

        c2.download_button(
            "Download CSV",
            data=read_history().to_csv(index=False),
            file_name="visionai_detection_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.dataframe(history.iloc[::-1], use_container_width=True, hide_index=True)
        st.caption("Snapshots are saved in the local unknown_faces/ directory.")

        if "Snapshot" in history.columns:
            valid = [
                Path(p) for p in history["Snapshot"].dropna().astype(str)
                if Path(p).exists()
            ]
            if valid:
                st.subheader("Recent snapshots")
                cols = st.columns(min(4, len(valid)))
                for i, image_path in enumerate(valid[-8:][::-1]):
                    cols[i % len(cols)].image(
                        str(image_path),
                        caption=image_path.name,
                        use_container_width=True,
                    )
