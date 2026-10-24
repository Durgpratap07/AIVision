from __future__ import annotations

import csv
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

FONT = cv2.FONT_HERSHEY_SIMPLEX


class VisionEngine:
    """Models, face database and per-frame inference for the Streamlit dashboard.

    Frames are pushed in by the streamlit-webrtc callback thread through
    process_frame(). The engine no longer opens the camera itself.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    MAX_WIDTH = 640               # frames wider than this are downscaled before processing
    STREAM_TIMEOUT_S = 2.0        # no frames for this long -> camera counts as stopped
    STALE_RESULT_S = 1.0          # cached boxes older than this are no longer drawn
    UNKNOWN_LOG_COOLDOWN_S = 3.0  # same cooldown as the original engine
    REFRESH_KNOWN_S = 2.0         # re-run SFace for an already-identified, tracked face
    REFRESH_UNKNOWN_S = 0.75      # retry sooner for faces currently labelled Unknown
    TRACK_IOU = 0.4               # box overlap needed to treat a face as the same track

    DEFAULT_SETTINGS = {
        "confidence": 0.35,
        "similarity_threshold": 0.363,
        "imgsz": 320,
        "yolo_interval": 3,
        "face_interval": 2,
        "detect_objects": True,
        "recognize_faces": True,
    }

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.known_dir = self.base_dir / "known_faces"
        self.models_dir = self.base_dir / "models"
        self.unknown_dir = self.base_dir / "unknown_faces"
        self.csv_path = self.base_dir / "unknown_log.csv"
        self.known_dir.mkdir(parents=True, exist_ok=True)
        self.unknown_dir.mkdir(parents=True, exist_ok=True)

        self.yolo_path = self.base_dir / "yolov8n.pt"
        self.yunet_path = self.models_dir / "face_detection_yunet_2023mar.onnx"
        self.sface_path = self.models_dir / "face_recognition_sface_2021dec.onnx"

        missing = [str(p) for p in (self.yolo_path, self.yunet_path, self.sface_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Required model file(s) missing:\n" + "\n".join(missing) +
                "\nCopy yolov8n.pt to the project root and the two ONNX files into models/."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo = YOLO(str(self.yolo_path))

        # Streaming models (used only by the WebRTC callback thread, via process_frame).
        self.face_detector, self.face_recognizer = self._make_face_models()
        # Separate instances for registration so uploads never block or race the live stream.
        self._reg_detector, self._reg_recognizer = self._make_face_models()
        self._reg_lock = threading.Lock()

        self.known_features: list[tuple[str, np.ndarray]] = []
        self.db_version = 0
        self.lock = threading.RLock()          # guards shared state read by the UI thread
        self._proc_lock = threading.Lock()     # serialises process_frame
        self._log_lock = threading.Lock()      # guards CSV appends

        self.settings = dict(self.DEFAULT_SETTINGS)

        # Streaming state (touched only while holding _proc_lock, except stats below)
        self._frame_idx = 0
        self._objects: list[tuple] = []
        self._objects_ts = 0.0
        self._tracks: list[dict] = []
        self._last_frame_ts = 0.0
        self.last_unknown_write = 0.0

        # Stats shared with the UI (guarded by self.lock)
        self.object_count = 0
        self.face_count = 0
        self.known_count = 0
        self.unknown_count = 0
        self.fps = 0.0
        self.yolo_ms = 0.0
        self.face_ms = 0.0
        self.last_error = ""
        self.last_events: list[dict] = []

        self.load_known_faces()
        self._ensure_csv()
        self._warmup()

    # ------------------------------------------------------------------ setup
    def _make_face_models(self):
        detector = cv2.FaceDetectorYN_create(
            str(self.yunet_path), "", (320, 320), 0.8, 0.3, 5000
        )
        recognizer = cv2.FaceRecognizerSF_create(str(self.sface_path), "")
        return detector, recognizer

    def _warmup(self):
        """Run one dummy inference so the first live frame does not stall."""
        try:
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.yolo.predict(dummy, device=self.device, imgsz=320, verbose=False)
            self.face_detector.setInputSize((320, 320))
            self.face_detector.detect(dummy)
        except Exception:
            pass

    def _ensure_csv(self):
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["Timestamp", "Name", "Snapshot", "Similarity"])

    def update_settings(self, **kwargs):
        """Thread-safe settings update; called from the UI thread, read by the callback."""
        with self.lock:
            for key, value in kwargs.items():
                if key in self.DEFAULT_SETTINGS:
                    self.settings[key] = value

    # ------------------------------------------------------- face database
    def _feature_from_image(self, image):
        h, w = image.shape[:2]
        self._reg_detector.setInputSize((w, h))
        _, faces = self._reg_detector.detect(image)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda item: item[2] * item[3])
        aligned = self._reg_recognizer.alignCrop(image, face)
        return self._reg_recognizer.feature(aligned)

    def load_known_faces(self):
        features = []
        with self._reg_lock:
            for person_dir in sorted(self.known_dir.iterdir()):
                if not person_dir.is_dir():
                    continue
                for image_path in sorted(person_dir.iterdir()):
                    if image_path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                        continue
                    image = cv2.imread(str(image_path))
                    if image is None:
                        continue
                    try:
                        feature = self._feature_from_image(image)
                        if feature is not None:
                            features.append((person_dir.name, feature))
                    except cv2.error:
                        continue
        with self.lock:
            self.known_features = features
            self.db_version += 1
        return len(features)

    def people(self):
        return sorted(p.name for p in self.known_dir.iterdir() if p.is_dir())

    def register_uploaded_images(self, person_name: str, uploaded_files):
        safe_name = "".join(c for c in person_name.strip() if c.isalnum() or c in " _-").strip()
        if not safe_name:
            raise ValueError("Enter a valid name.")
        folder = self.known_dir / safe_name
        folder.mkdir(parents=True, exist_ok=True)
        saved = 0
        for index, uploaded in enumerate(uploaded_files, start=1):
            data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                continue
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{index}.jpg"
            target = folder / filename
            if cv2.imwrite(str(target), image):
                saved += 1
        self.load_known_faces()
        return safe_name, saved

    def delete_person(self, name: str):
        folder = self.known_dir / name
        if folder.exists() and folder.is_dir():
            for item in folder.iterdir():
                if item.is_file():
                    item.unlink()
            folder.rmdir()
            self.load_known_faces()

    # ------------------------------------------------------ matching helpers
    def _best_match(self, feature, references):
        """Identical logic to the original engine: best cosine score over all references."""
        best_name = "Unknown"
        best_score = -1.0
        for known_name, known_feature in references:
            score = float(self.face_recognizer.match(
                feature, known_feature, cv2.FaceRecognizerSF_FR_COSINE
            ))
            if score > best_score:
                best_score = score
                best_name = known_name
        return best_name, best_score

    @staticmethod
    def _resolve(raw_name, score, threshold):
        if raw_name == "Unknown" or score < threshold:
            return "Unknown"
        return raw_name

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    # ------------------------------------------------------------ inference
    def _run_yolo(self, frame, s):
        result = self.yolo.predict(
            frame, device=self.device, conf=s["confidence"], imgsz=int(s["imgsz"]), verbose=False
        )[0]
        objects = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            names = result.names
            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
                objects.append((int(x1), int(y1), int(x2), int(y2),
                                str(names[int(cls)]), float(conf), int(cls)))
        return objects

    def _run_faces(self, frame, s, now):
        h, w = frame.shape[:2]
        self.face_detector.setInputSize((w, h))
        _, faces = self.face_detector.detect(frame)
        with self.lock:
            references = self.known_features
            version = self.db_version
        threshold = s["similarity_threshold"]

        unused = list(self._tracks)
        tracks = []
        for face in (faces if faces is not None else []):
            box = tuple(int(v) for v in face[:4])
            match, best_iou = None, 0.0
            for track in unused:
                iou = self._iou(box, track["box"])
                if iou > best_iou:
                    match, best_iou = track, iou
            if match is not None and best_iou >= self.TRACK_IOU:
                unused.remove(match)
                track = match
                track["box"] = box
                known = self._resolve(track["raw"], track["score"], threshold) != "Unknown"
                limit = self.REFRESH_KNOWN_S if known else self.REFRESH_UNKNOWN_S
                needs_recognition = (now - track["t_recog"] >= limit) or track["version"] != version
            else:
                track = {"box": box, "raw": "Unknown", "score": -1.0, "t_recog": 0.0, "version": version}
                needs_recognition = True
            if needs_recognition:
                aligned = self.face_recognizer.alignCrop(frame, face)
                feature = self.face_recognizer.feature(aligned)
                track["raw"], track["score"] = self._best_match(feature, references)
                track["t_recog"] = now
                track["version"] = version
            track["seen"] = now
            tracks.append(track)
        self._tracks = tracks

    # --------------------------------------------------------------- drawing
    @staticmethod
    def _class_color(cls):
        return ((cls * 67 + 40) % 200 + 55, (cls * 131 + 90) % 200 + 55, (cls * 29 + 150) % 200 + 55)

    def _draw_objects(self, frame, objects):
        for x1, y1, x2, y2, name, conf, cls in objects:
            color = self._class_color(cls)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
            top = max(0, y1 - th - 8)
            cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + 8), color, -1)
            cv2.putText(frame, label, (x1 + 3, top + th + 2), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_faces(self, frame, tracks, threshold):
        h, w = frame.shape[:2]
        known_count = unknown_count = 0
        unknown_scores = []
        for track in tracks:
            x, y, fw, fh = track["box"]
            best_score = track["score"]
            best_name = self._resolve(track["raw"], best_score, threshold)
            is_known = best_name != "Unknown"
            color = (40, 210, 100) if is_known else (40, 60, 240)
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
            label = f"{best_name}" + (f"  {best_score:.2f}" if best_score >= 0 else "")
            cv2.rectangle(frame, (x, max(0, y - 30)),
                          (min(w - 1, x + max(150, len(label) * 10)), y), color, -1)
            cv2.putText(frame, label, (x + 5, max(18, y - 9)), FONT, 0.55, (255, 255, 255), 2)
            if is_known:
                known_count += 1
            else:
                unknown_count += 1
                unknown_scores.append(best_score)
        return known_count, unknown_count, unknown_scores

    # ------------------------------------------------------------- main entry
    def process_frame(self, frame):
        """Annotate one BGR frame. Called from the streamlit-webrtc callback thread.

        If a previous frame is still being processed, the new frame is passed
        through unannotated instead of queueing, which keeps video real-time.
        """
        if not self._proc_lock.acquire(blocking=False):
            return frame
        try:
            return self._process(frame)
        finally:
            self._proc_lock.release()

    def _process(self, frame):
        now = time.time()
        with self.lock:
            s = dict(self.settings)

        h, w = frame.shape[:2]
        if w > self.MAX_WIDTH:
            scale = self.MAX_WIDTH / w
            frame = cv2.resize(frame, (self.MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)

        if now - self._last_frame_ts > self.STREAM_TIMEOUT_S:   # stream (re)started
            self._frame_idx = 0
            self._objects, self._objects_ts, self._tracks = [], 0.0, []
        gap = now - self._last_frame_ts if self._last_frame_ts else 0.0
        self._last_frame_ts = now
        self._frame_idx += 1
        first_of_cycle = self._frame_idx - 1

        error = ""
        yolo_ms = face_ms = None

        # ---- object detection (every Nth frame, otherwise reuse cached boxes)
        if s["detect_objects"]:
            if first_of_cycle % max(1, int(s["yolo_interval"])) == 0:
                try:
                    t0 = time.perf_counter()
                    self._objects = self._run_yolo(frame, s)
                    self._objects_ts = now
                    yolo_ms = (time.perf_counter() - t0) * 1000
                except Exception as exc:
                    error = f"Object detection error: {exc}"
        else:
            self._objects = []

        # ---- face detection + recognition (own interval, tracked identities reused)
        ran_faces = False
        if s["recognize_faces"]:
            if first_of_cycle % max(1, int(s["face_interval"])) == 0:
                try:
                    t0 = time.perf_counter()
                    self._run_faces(frame, s, now)
                    face_ms = (time.perf_counter() - t0) * 1000
                    ran_faces = True
                except Exception as exc:
                    error = f"Face recognition error: {exc}"
        else:
            self._tracks = []

        # ---- draw cached results (drop anything stale)
        objects = self._objects if now - self._objects_ts <= self.STALE_RESULT_S else []
        tracks = [t for t in self._tracks if now - t["seen"] <= self.STALE_RESULT_S]
        self._draw_objects(frame, objects)
        known_count, unknown_count, unknown_scores = self._draw_faces(
            frame, tracks, s["similarity_threshold"]
        )

        # ---- unknown-face logging (background thread so disk I/O never stalls video)
        if ran_faces and unknown_scores and now - self.last_unknown_write >= self.UNKNOWN_LOG_COOLDOWN_S:
            self.last_unknown_write = now
            threading.Thread(
                target=self._log_unknown, args=(frame.copy(), unknown_scores[0]), daemon=True
            ).start()

        with self.lock:
            self.object_count = len(objects)
            self.face_count = len(tracks)
            self.known_count = known_count
            self.unknown_count = unknown_count
            if gap > 0:
                inst = 1.0 / gap
                self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
            if yolo_ms is not None:
                self.yolo_ms = yolo_ms if self.yolo_ms == 0 else 0.7 * self.yolo_ms + 0.3 * yolo_ms
            if face_ms is not None:
                self.face_ms = face_ms if self.face_ms == 0 else 0.7 * self.face_ms + 0.3 * face_ms
            if error:
                self.last_error = error
            elif ran_faces or yolo_ms is not None:
                self.last_error = ""
        return frame

    # --------------------------------------------------------------- logging
    def _log_unknown(self, frame, similarity):
        try:
            stamp = datetime.now()
            filename = f"unknown_{stamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            path = self.unknown_dir / filename
            if cv2.imwrite(str(path), frame):
                with self._log_lock:
                    with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(
                            [stamp.isoformat(timespec="seconds"), "Unknown", str(path), similarity]
                        )
                with self.lock:
                    self.last_events.insert(0, {
                        "timestamp": stamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "name": "Unknown",
                        "snapshot": str(path),
                        "similarity": round(float(similarity), 3),
                    })
                    self.last_events = self.last_events[:100]
        except Exception as exc:
            with self.lock:
                self.last_error = f"Unknown-face logging error: {exc}"

    # -------------------------------------------------------------- UI state
    def snapshot(self):
        with self.lock:
            running = (time.time() - self._last_frame_ts) < self.STREAM_TIMEOUT_S
            return {
                "running": running,
                "object_count": self.object_count if running else 0,
                "face_count": self.face_count if running else 0,
                "known_count": self.known_count if running else 0,
                "unknown_count": self.unknown_count if running else 0,
                "fps": self.fps if running else 0.0,
                "yolo_ms": self.yolo_ms,
                "face_ms": self.face_ms,
                "last_error": self.last_error,
                "registered_images": len(self.known_features),
                "people": self.people(),
                "events": list(self.last_events),
                "settings": dict(self.settings),
            }