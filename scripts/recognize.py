import sys
import os
import csv
from datetime import datetime
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from mtcnn import MTCNN
from model.embedding_model import load_embedding_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

import win32gui
import win32con

SAVE_DIR = 'embeddings'
THRESHOLD = 0.85
HISTORY_PATH = 'history.csv'
USERS_JSON_PATH = os.path.join(SAVE_DIR, 'users.json')

def get_user_id(name):
    """Lấy ID từ users.json dựa trên tên."""
    if os.path.exists(USERS_JSON_PATH):
        try:
            with open(USERS_JSON_PATH, 'r', encoding='utf-8') as f:
                users = json.load(f)
                for user in users:
                    if user['name'] == name:
                        return user['id']
        except json.JSONDecodeError:
            print(f"Warning: {USERS_JSON_PATH} is corrupted.")
    return "Unknown"

def log_attendance(name, status):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_id = get_user_id(name)
    with open(HISTORY_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([user_id, name, now, status])

def load_all_embeddings():
    vector_path = os.path.join(SAVE_DIR, 'vectors.npy')
    name_path = os.path.join(SAVE_DIR, 'names.npy')
    new_vector_path = os.path.join(SAVE_DIR, 'new_vectors.npy')
    new_name_path = os.path.join(SAVE_DIR, 'new_names.npy')

    vectors, names = [], []

    if os.path.exists(vector_path) and os.path.exists(name_path):
        vectors.append(np.load(vector_path))
        names.append(np.load(name_path))

    if os.path.exists(new_vector_path) and os.path.exists(new_name_path):
        try:
            if os.path.getsize(new_vector_path) > 0 and os.path.getsize(new_name_path) > 0:
                new_vectors = np.load(new_vector_path)
                new_names = np.load(new_name_path)
                vectors.append(new_vectors)
                names.append(new_names)
            else:
                print("[!] File new_embeddings tồn tại nhưng trống, bỏ qua.")
        except Exception as e:
            print(f"[!] Không thể load new_embeddings: {e}")

    if vectors and names:
        vectors = np.vstack(vectors)
        names = np.concatenate(names)
        return names, vectors

    return np.array([]), np.array([])

def recognize_faces(status):
    model = load_embedding_model()
    detector = MTCNN()
    names, vectors = load_all_embeddings()

    vectors_loaded = len(names) > 0  

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    WINDOW_NAME = "Face Recognition"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    hwnd = win32gui.FindWindow(None, WINDOW_NAME)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 165, 150, 650, 500, 0)

    recognized = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)

        if len(faces) == 0:
            cv2.putText(frame, "No face detected!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            face = faces[0]
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            face_img = frame[y:y+h, x:x+w]

            try:
                face_img = cv2.resize(face_img, (224, 224))
            except:
                continue

            face_array = preprocess_input(img_to_array(face_img))
            face_array = np.expand_dims(face_array, axis=0)

            emb = model.predict(face_array)[0]

            name = "Unknown"
            color = (0, 0, 255)
            notification = "Failed. Try again!!!"

            if vectors_loaded:
                sims = cosine_similarity([emb], vectors)[0]
                best_idx = np.argmax(sims)
                best_score = sims[best_idx]

                if best_score >= THRESHOLD:
                    name = names[best_idx]
                    notification = "Successful"
                    color = (0, 255, 0)

                    if name not in recognized:
                        log_attendance(name, status)
                        recognized.add(name)

            # Vẽ kết quả
            cv2.putText(frame, notification, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    status = sys.argv[1] if len(sys.argv) > 1 else "Check-in"
    recognize_faces(status)