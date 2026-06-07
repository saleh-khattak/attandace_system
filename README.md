# 🎓 Attendance System Using Face Recognition

A real-time attendance management system built with Python and OpenCV.
Automatically detects and recognizes faces via webcam, marks attendance,
and exports records to a running Excel log.

---

## 📸 Features

- Real-time multi-face detection and recognition
- Automatic attendance marking with timestamp
- Duplicate prevention — each person marked once per day
- Persistent Excel log across multiple sessions
- Easily scalable — add new people anytime without rebuilding from scratch

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| OpenCV | Webcam access, face detection, real-time display |
| LBPH Face Recognizer | Face recognition (opencv-contrib) |
| Haar Cascade | Face detection (pre-trained XML model) |
| Pandas | Building and managing attendance records |
| OpenPyXL | Reading and writing Excel files |

---

## 📁 Project Structure
attendance_system/
├── dataset/                        # Face images per person (gitignored)
│   ├── Person Name/
│   │   ├── 1.jpg ... 300.jpg
├── trainer/
│   ├── trainer.yml                 # Trained LBPH model (gitignored)
│   └── labels.json                 # Label ID → Name mapping
├── attendance/
│   └── attendance_log.xlsx         # Running Excel attendance log (gitignored)
├── 01_collect_faces.py             # Step 1 — Collect face images
├── 02_train_model.py               # Step 2 — Train the recognizer
├── 03_recognize_attend.py          # Step 3 — Live attendance
├── requirements.txt
├── .gitignore
└── README.md

Install all at once:
```bash
pip install -r requirements.txt
```

---

## ⚠️ Notes

- `opencv-contrib-python` is required — it contains the LBPH Face Recognizer
- Make sure only one of `opencv-python` or `opencv-contrib-python` is installed, not both
- Good lighting during face collection improves recognition accuracy significantly
- Recommended: 300 samples per person with slight head movement during collection

---

## 👤 Author

**Muhammad Saleh**  
AI & Data Science Portfolio Project  
[GitHub](https://github.com/saleh-khattak) • [LinkedIn](www.linkedin.com/in/muhammad-saleh-a842b434a)