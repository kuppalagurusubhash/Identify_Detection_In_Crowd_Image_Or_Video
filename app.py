import os
import cv2
import json
import uuid
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response, send_file
from werkzeug.utils import secure_filename
from datetime import datetime
import csv
import io

from utils.database import (
    init_db, add_person, add_face_embedding, get_all_persons,
    delete_person, get_detection_history, get_dashboard_stats, log_detection
)
from services.face_detector import FaceDetector
from services.embedding_service import EmbeddingService
from services.matcher import IdentityMatcher
from services.video_processor import VideoProcessor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ai-crowd-identity-detection-secret-2026'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'outputs')
DATASET_FOLDER = os.path.join(BASE_DIR, 'datasets', 'registered_faces')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'mp4', 'avi', 'mov', 'mkv'}

# Initialize Computer Vision Engines
detector = FaceDetector()
embedder = EmbeddingService()
matcher = IdentityMatcher()
video_processor = VideoProcessor(frame_sample_rate=3)

# Global camera reference for live webcam feed
camera = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.before_request
def initialize_system():
    init_db()

@app.route('/')
def dashboard():
    stats = get_dashboard_stats()
    return render_template('index.html', stats=stats)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        registration_id = request.form.get('registration_id', '').strip()
        department = request.form.get('department', '').strip()

        if not name or not registration_id or not department:
            return jsonify({'success': False, 'message': 'All fields (Name, Reg ID, Dept) are required.'}), 400

        files = request.files.getlist('photos')
        if not files or files[0].filename == '':
            return jsonify({'success': False, 'message': 'Please upload at least one face photo.'}), 400

        # Save person to DB
        person_id = add_person(name, registration_id, department)
        if not person_id:
            return jsonify({'success': False, 'message': f'Registration ID {registration_id} already exists.'}), 400

        person_dir = os.path.join(DATASET_FOLDER, secure_filename(name))
        os.makedirs(person_dir, exist_ok=True)

        embeddings_created = 0
        for file in files:
            if file and allowed_file(file.filename):
                filename = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
                filepath = os.path.join(person_dir, filename)
                file.save(filepath)

                # Process face image
                img = cv2.imread(filepath)
                if img is not None:
                    faces = detector.detect_faces(img)
                    if faces:
                        # Use first/best face crop
                        crop = faces[0]['crop']
                        emb = embedder.get_embedding(crop)
                        if emb is not None:
                            rel_path = os.path.relpath(filepath, BASE_DIR).replace('\\', '/')
                            add_face_embedding(person_id, emb, rel_path)
                            embeddings_created += 1
                    else:
                        # Fallback: use whole image if no bounding box found
                        emb = embedder.get_embedding(img)
                        if emb is not None:
                            rel_path = os.path.relpath(filepath, BASE_DIR).replace('\\', '/')
                            add_face_embedding(person_id, emb, rel_path)
                            embeddings_created += 1

        if embeddings_created > 0:
            return jsonify({
                'success': True,
                'message': f'Successfully registered {name} with {embeddings_created} facial sample(s).'
            })
        else:
            return jsonify({'success': False, 'message': 'Could not extract face embeddings from uploaded images.'}), 400

    return render_template('register.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filename = f"crowd_{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        image = cv2.imread(input_path)
        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        faces = detector.detect_faces(image)
        results = []
        recognized_count = 0
        unknown_count = 0

        annotated_img = image.copy()

        for idx, face in enumerate(faces):
            x, y, w, h = face['box']
            crop = face['crop']

            emb = embedder.get_embedding(crop)
            match = matcher.match_embedding(emb)

            if match['is_match']:
                recognized_count += 1
                color = (0, 220, 100) # Bright Emerald Green
                label = f"{match['name']} ({int(match['similarity']*100)}%)"
            else:
                unknown_count += 1
                color = (0, 70, 255) # Crimson Red / Orange
                label = f"Unknown ({int(match['similarity']*100)}%)"

            # Draw bounding box on image
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 3)

            # Draw pill label header
            (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_img, (x, max(0, y - txt_h - 10)), (x + txt_w + 12, y), color, -1)
            cv2.putText(annotated_img, label, (x + 6, max(txt_h, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            # Log to DB
            log_detection(
                person_id=match['person_id'],
                person_name=match['name'],
                source_type='image',
                source_name=filename,
                confidence=match['similarity']
            )

            results.append({
                'face_id': idx + 1,
                'box': [x, y, w, h],
                'name': match['name'],
                'registration_id': match['registration_id'],
                'department': match['department'],
                'similarity': match['similarity'],
                'is_match': match['is_match']
            })

        output_filename = f"annotated_{filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_path, annotated_img)

        relative_output_url = f"/static/outputs/{output_filename}"

        return jsonify({
            'success': True,
            'image_url': relative_output_url,
            'total_detected': len(faces),
            'recognized_count': recognized_count,
            'unknown_count': unknown_count,
            'detections': results
        })

    return render_template('detect.html')

@app.route('/video', methods=['GET', 'POST'])
def detect_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        file = request.files['video']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid video file format'}), 400

        filename = f"video_{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        output_filename = f"processed_{filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        summary = video_processor.process_video(input_path, output_path, source_name=filename)
        relative_video_url = f"/static/outputs/{output_filename}"

        return jsonify({
            'success': True,
            'video_url': relative_video_url,
            'summary': summary
        })

    return render_template('video.html')

def generate_webcam_frames():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Process frame for real-time live identification
            faces = detector.detect_faces(frame)
            for face in faces:
                x, y, w, h = face['box']
                crop = face['crop']
                emb = embedder.get_embedding(crop)
                match = matcher.match_embedding(emb)

                if match['is_match']:
                    color = (0, 255, 127)
                    label = f"{match['name']} ({int(match['similarity']*100)}%)"
                else:
                    color = (0, 80, 255)
                    label = f"Unknown ({int(match['similarity']*100)}%)"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, y - txt_h - 8), (x + txt_w + 10, y), color, -1)
                cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/live')
def live_camera():
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_webcam_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/persons')
def persons_page():
    persons = get_all_persons()
    return render_template('persons.html', persons=persons)

@app.route('/api/delete_person/<int:person_id>', methods=['POST'])
def api_delete_person(person_id):
    delete_person(person_id)
    return jsonify({'success': True, 'message': 'Person deleted successfully.'})

@app.route('/history')
def history_page():
    source_type = request.args.get('source_type', None)
    history = get_detection_history(limit=200, source_type=source_type)
    return render_template('history.html', history=history, current_filter=source_type)

@app.route('/api/export_history')
def export_history():
    history = get_detection_history(limit=1000)
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Detection ID', 'Person Name', 'Source Type', 'Source Name', 'Similarity Score', 'Timestamp'])
    
    for row in history:
        writer.writerow([
            row['detection_id'],
            row['person_name'],
            row['source_type'],
            row['source_name'],
            row['confidence'],
            row['timestamp']
        ])
    
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment;filename=detection_history.csv'}
    )

if __name__ == '__main__':
    print("Starting AI Crowd Identity Detection & Recognition Server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
