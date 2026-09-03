<<<<<<< HEAD
# Identify_Detection_In_Crowd_Image_Or_Video
=======
# AI-Based Identity Detection and Recognition in Crowd Images and Video Streams

An AI-powered computer vision and deep learning web application designed to detect, recognize, and track multiple individuals in crowd images, recorded video streams, and live webcam feeds.

## 🚀 Key Features

- **Multi-Person Face Detection**: Utilizes OpenCV multi-scale cascades and adaptive contour extraction with Non-Maximum Suppression (NMS).
- **Deep Facial Embedding Extraction**: Extracts 576-dimensional L2-normalized feature vectors using PyTorch MobileNetV3 deep learning backbone.
- **Cosine Similarity Matching Engine**: Matches query face embeddings against an authorized identity database with tuned thresholding ($S \ge 0.60$) and multi-sample score aggregation.
- **Crowd Video Stream Analysis**: Frame-by-frame analysis with temporal tracking, bounding box annotations, and output MP4 generation.
- **Real-Time Live Camera Feed**: Live streaming webcam feed via Flask MJPEG stream (`/video_feed`) with instant identity bounding boxes and similarity scores.
- **Glassmorphic Web Dashboard**: Dark theme dashboard built with HTML5, CSS3, JavaScript, Chart.js, and Flask.

## 📁 Repository Structure

```
crowd-identity-detection/
├── app.py                      # Flask REST API & Web Application server
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── database/                   # SQLite database storage (identity.db)
├── services/
│   ├── face_detector.py        # Face detection & bounding box extraction
│   ├── embedding_service.py    # PyTorch MobileNetV3 deep feature extractor
│   ├── matcher.py              # Cosine similarity matching engine
│   └── video_processor.py      # Video frame processing & MP4 generator
├── utils/
│   ├── database.py             # SQLite CRUD helper functions
│   └── preprocessing.py        # CLAHE lighting enhancement & crop normalization
├── static/
│   ├── css/style.css           # Glassmorphism dark mode stylesheet
│   └── outputs/                # Processed images and video outputs
└── templates/                  # Jinja2 HTML templates
    ├── base.html
    ├── index.html              # Dashboard homepage
    ├── register.html           # Identity registration page
    ├── detect.html             # Crowd image detection page
    ├── video.html              # Crowd video processing page
    ├── live.html               # Live webcam feed page
    ├── persons.html            # Registered database management page
    └── history.html            # Detection activity logs page
```

## ⚙️ Installation & Usage

1. **Clone Repository**:
   ```bash
   git clone https://github.com/kuppalagurusubhash/Identify_Detection_In_Crowd_Image_Or_Video.git
   cd Identify_Detection_In_Crowd_Image_Or_Video
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Application**:
   ```bash
   python app.py
   ```

4. **Access Web App**:
   Open browser at `http://127.0.0.1:5000`
>>>>>>> a0e521b (feat: complete AI crowd identity detection and recognition system MVP)
