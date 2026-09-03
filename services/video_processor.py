import os
import cv2
import numpy as np
from datetime import datetime
from services.face_detector import FaceDetector
from services.embedding_service import EmbeddingService
from services.matcher import IdentityMatcher
from utils.database import log_detection

class VideoProcessor:
    def __init__(self, frame_sample_rate=3):
        self.detector = FaceDetector()
        self.embedder = EmbeddingService()
        self.matcher = IdentityMatcher()
        self.frame_sample_rate = frame_sample_rate

    def process_video(self, input_video_path, output_video_path, source_name="video"):
        """
        Reads input video frame by frame, detects/identifies crowd faces,
        draws bounding boxes & labels, logs detections, and writes output MP4.
        """
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video file not found: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {input_video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # FourCC codec for MP4 video output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_count = 0
        processed_count = 0
        total_faces_detected = 0
        matched_faces_count = 0
        unknown_faces_count = 0

        # Memory tracker for temporal smoothing between sampled frames
        cached_detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Perform detection & recognition every Nth frame
            if frame_count % self.frame_sample_rate == 0 or frame_count == 1:
                processed_count += 1
                faces = self.detector.detect_faces(frame)
                cached_detections = []

                for face in faces:
                    x, y, w, h = face['box']
                    crop = face['crop']
                    emb = self.embedder.get_embedding(crop)
                    match_res = self.matcher.match_embedding(emb)

                    cached_detections.append({
                        'box': (x, y, w, h),
                        'match': match_res
                    })

                    total_faces_detected += 1
                    if match_res['is_match']:
                        matched_faces_count += 1
                    else:
                        unknown_faces_count += 1

                    # Log detection to SQLite DB
                    log_detection(
                        person_id=match_res['person_id'],
                        person_name=match_res['name'],
                        source_type='video',
                        source_name=source_name,
                        confidence=match_res['similarity']
                    )

            # Draw cached annotations on frame
            for det in cached_detections:
                x, y, w, h = det['box']
                match = det['match']

                if match['is_match']:
                    color = (0, 255, 127)  # Vibrant Emerald Green for recognized matches
                    label = f"{match['name']} ({int(match['similarity']*100)}%)"
                else:
                    color = (0, 80, 255)   # Vivid Red/Orange for Unknown
                    label = f"Unknown ({int(match['similarity']*100)}%)"

                # Bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Header text pill
                (txt_w, txt_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, y - txt_h - 8), (x + txt_w + 10, y), color, -1)
                cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            out.write(frame)

        cap.release()
        out.release()

        return {
            'total_frames': total_frames,
            'processed_frames': processed_count,
            'total_faces': total_faces_detected,
            'recognized_faces': matched_faces_count,
            'unknown_faces': unknown_faces_count,
            'output_path': output_video_path
        }
