
# Real-Time Object and Face Recognition using YOLOv8 🎯🧠

A Python-based real-time object and face recognition system using the **YOLOv8** deep learning model and `face_recognition`, integrated with **OpenCV** for webcam-based live detection. The project runs on GPU (CUDA) for faster performance and supports logging of unknown faces along with periodic snapshots.

---
## Object Detection and Face Recognition through real time footage with yolo and pytorch
## 🔍 Features

- ✅ Real-time object detection using YOLOv8 (pre-trained on COCO dataset)
- ✅ Real-time face recognition with custom known faces
- ✅ Logs unknown faces every 2 seconds with snapshot storage
- ✅ Supports GPU acceleration (CUDA)
- ✅ CSV logging of unknown detections with timestamps
- ✅ Clean bounding box overlays with labels and confidence scores

---

## 🚀 Technologies Used

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [NumPy](https://numpy.org/)
- Python 3.10+

---

## 📂 Folder Structure

- project_root/
- │
- ├── detect.py # Main script
- ├── known_faces/ # Folder with subfolders for each known person
- │ └── Alice/
- │ ├── alice1.jpg
- │ └── alice2.jpg
- ├── snapshots/ # Automatically stores unknown face images
- ├── unknown_log.csv # CSV log of unknown detections
- ├── yolo-Weights/
- │ └── yolov8n.pt # YOLOv8 model file
- └── README.md

## 💾 Installation

1. **Clone the repository**
     ```bash
     git clone https://github.com/your-username/your-repo-name.git
     cd your-repo-name
2. **Set up a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
3. **Install dependencies**
   ```bash
    pip install -r requirements.txt
4. **Download YOLOv8n weights**
     ```bash
     Place yolov8n.pt inside yolo-Weights/. You can get it from here.
5. **Run the detection script**
   ```bash
   python detect.py
   
📸 Sample Output
<img width="1200" height="560" alt="Screenshot 2025-07-13 165849" src="https://github.com/user-attachments/assets/6d2caf7a-19d8-43ba-9af3-b838961c5734" />

🛠️ Future Improvements

 Sound alert on unknown detection (optional)

 Option to trigger snapshot for known users too

 Dashboard UI for visual logs

 Optimize for edge devices or Raspberry Pi

 Integrate TrOCR or handwriting recognition for diary/logs

👤 Author

Sai Vinod
- Feel free to connect or reach out on GitHub for collaborations or questions!

📄 License

This project is licensed under the MIT License.
