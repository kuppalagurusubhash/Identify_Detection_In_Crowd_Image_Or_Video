import os
import sqlite3
import json
from datetime import datetime, timedelta

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database')
DB_PATH = os.path.join(DB_DIR, 'identity.db')

def get_db_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Persons Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            person_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            registration_id TEXT UNIQUE NOT NULL,
            department TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Face Embeddings Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_embeddings (
            embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            embedding TEXT NOT NULL,
            image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES persons (person_id) ON DELETE CASCADE
        )
    ''')

    # Detections Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            person_name TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

def add_person(name, registration_id, department):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO persons (name, registration_id, department)
            VALUES (?, ?, ?)
        ''', (name, registration_id, department))
        person_id = cursor.lastrowid
        conn.commit()
        return person_id
    except sqlite3.IntegrityError:
        # Person already exists with this registration_id, fetch id
        cursor.execute('SELECT person_id FROM persons WHERE registration_id = ?', (registration_id,))
        row = cursor.fetchone()
        return row['person_id'] if row else None
    finally:
        conn.close()

def add_face_embedding(person_id, embedding_vector, image_path):
    conn = get_db_connection()
    cursor = conn.cursor()
    embedding_json = json.dumps([float(x) for x in embedding_vector])
    cursor.execute('''
        INSERT INTO face_embeddings (person_id, embedding, image_path)
        VALUES (?, ?, ?)
    ''', (person_id, embedding_json, image_path))
    embedding_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return embedding_id

def get_all_embeddings():
    """
    Returns list of dicts:
    [{'person_id': 1, 'name': 'Rahul', 'registration_id': 'REV001', 'department': 'CSE', 'embedding': [floats...], 'image_path': '...'}]
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT p.person_id, p.name, p.registration_id, p.department, fe.embedding, fe.image_path
        FROM face_embeddings fe
        JOIN persons p ON fe.person_id = p.person_id
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        results.append({
            'person_id': r['person_id'],
            'name': r['name'],
            'registration_id': r['registration_id'],
            'department': r['department'],
            'embedding': json.loads(r['embedding']),
            'image_path': r['image_path']
        })
    return results

def get_all_persons():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT p.person_id, p.name, p.registration_id, p.department, p.created_at,
               COUNT(fe.embedding_id) as sample_count
        FROM persons p
        LEFT JOIN face_embeddings fe ON p.person_id = fe.person_id
        GROUP BY p.person_id
        ORDER BY p.created_at DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_person(person_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM persons WHERE person_id = ?', (person_id,))
    conn.commit()
    conn.close()

def log_detection(person_id, person_name, source_type, source_name, confidence):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO detections (person_id, person_name, source_type, source_name, confidence)
        VALUES (?, ?, ?, ?, ?)
    ''', (person_id, person_name, source_type, source_name, confidence))
    conn.commit()
    conn.close()

def get_detection_history(limit=100, source_type=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if source_type:
        cursor.execute('''
            SELECT * FROM detections WHERE source_type = ? ORDER BY timestamp DESC LIMIT ?
        ''', (source_type, limit))
    else:
        cursor.execute('''
            SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?
        ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_dashboard_stats():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) as total FROM persons')
    total_registered = cursor.fetchone()['total']

    cursor.execute('SELECT COUNT(*) as total FROM detections')
    total_detections = cursor.fetchone()['total']

    # Today's detections
    today_str = datetime.now().strftime('%Y-%m-%d')
    cursor.execute("SELECT COUNT(*) as total FROM detections WHERE timestamp LIKE ?", (f"{today_str}%",))
    detections_today = cursor.fetchone()['total']

    # Identified vs Unknown
    cursor.execute("SELECT COUNT(*) as total FROM detections WHERE person_name != 'Unknown'")
    total_identified = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(*) as total FROM detections WHERE person_name = 'Unknown'")
    total_unknown = cursor.fetchone()['total']

    # Detections by source type
    cursor.execute("SELECT source_type, COUNT(*) as count FROM detections GROUP BY source_type")
    by_source = {row['source_type']: row['count'] for row in cursor.fetchall()}

    # Recent 5 detections
    cursor.execute("SELECT * FROM detections ORDER BY timestamp DESC LIMIT 5")
    recent_detections = [dict(r) for r in cursor.fetchall()]

    conn.close()

    match_rate = round((total_identified / total_detections * 100), 1) if total_detections > 0 else 0.0

    return {
        'total_registered': total_registered,
        'total_detections': total_detections,
        'detections_today': detections_today,
        'total_identified': total_identified,
        'total_unknown': total_unknown,
        'match_rate': match_rate,
        'by_source': by_source,
        'recent_detections': recent_detections
    }
