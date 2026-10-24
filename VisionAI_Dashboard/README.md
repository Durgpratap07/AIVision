# VisionAI Streamlit Dashboard

A local Windows dashboard for the existing YOLOv8 + OpenCV YuNet/SFace project.

## Project layout

Copy these files into your existing `D:\Ai vision` folder. Keep your existing `yolov8n.pt` file.
Put the two ONNX files inside `models\`:
- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`

The dashboard creates `known_faces\`, `unknown_faces\`, and `unknown_log.csv` if needed.

## Install

Open PowerShell in the project folder. Use the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt
```

If pip reports a conflict with a different OpenCV package, remove both OpenCV wheels first and reinstall the required one:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y opencv-python opencv-contrib-python
.\.venv\Scripts\python.exe -m pip install opencv-contrib-python==4.9.0.80
```

Do not uninstall/reinstall torch unless it is actually missing; the existing CPU torch setup is fine.

## Launch

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open it in the browser on the same laptop.

## Pages
- Overview: current-frame counters and recent unknown-face events.
- Live Monitoring: start/stop webcam, set object confidence and face similarity threshold.
- Face Management: upload reference images, view stored images, delete a person's registration.
- Detection History: browse and download the unknown-face CSV log and inspect snapshots.

## Notes
- Webcam access is from the Python process running locally, not from a remote Streamlit server.
- Do not run the old `detect.py` and the dashboard at the same time; both may compete for the webcam.
- Initial implementation uses a background camera thread and Streamlit reruns to refresh the live feed. If the feed feels choppy on a CPU laptop, reduce webcam resolution or increase the refresh interval.
- YuNet/SFace similarity thresholds vary with image quality, pose and lighting. The default 0.363 is a starting point, not a reliable security-grade identity guarantee.
- This is a local demo. Use consented reference images and protect face snapshots/logs.
