import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from mtcnn import MTCNN
from model.embedding_model import load_embedding_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import win32gui
import json
import win32con

# Khởi tạo mô hình một lần duy nhất
detector = MTCNN()
model = load_embedding_model()

SAVE_DIR = 'embeddings'
CAPTURED_IMAGES_BASE_DIR = 'captured_images'
NUM_IMAGES = 100
IMG_SIZE = (224, 224)
USERS_JSON_PATH = os.path.join(SAVE_DIR, 'users.json')

if not os.path.exists(CAPTURED_IMAGES_BASE_DIR):
    os.makedirs(CAPTURED_IMAGES_BASE_DIR)

def capture_images(person_id, name):
    try:
        person_dir = os.path.join(CAPTURED_IMAGES_BASE_DIR, f"{person_id}_{name}")
        
        # Kiểm tra trùng lặp ID trong users.json
        if os.path.exists(USERS_JSON_PATH):
            try:
                with open(USERS_JSON_PATH, 'r', encoding='utf-8') as f:
                    users = json.load(f)
                    if any(user['id'] == person_id for user in users):
                        print(f"Error: Person with ID '{person_id}' already exists in users.json.")
                        return
            except json.JSONDecodeError:
                print(f"Error: {USERS_JSON_PATH} is corrupted.")
                return
            except Exception as e:
                print(f"Error: Failed to read {USERS_JSON_PATH}: {e}")
                return

        # Kiểm tra trùng lặp thư mục
        if os.path.exists(person_dir):
            print(f"Error: Person with ID '{person_id}' and name '{name}' already exists in captured images.")
            return

        embeddings = []
        os.makedirs(person_dir)

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Sử dụng CAP_DSHOW để giảm độ trễ
        if not cap.isOpened():
            print("Error: Cannot open camera.")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Cấu hình cửa sổ
        cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Capture", 640, 480)
        cv2.moveWindow("Capture", 160, 120)

        hwnd = win32gui.FindWindow(None, "Capture")
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 165, 150, 650, 500, 0)

        count = 0
        frame_count = 0
        last_face_box = None
        print(f"Capturing images for: {name}")

        while count < NUM_IMAGES:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Chỉ phát hiện khuôn mặt mỗi 5 khung hình
            if frame_count % 5 == 0 or last_face_box is None:
                faces = detector.detect_faces(frame)
            else:
                faces = [last_face_box] if last_face_box else []

            if len(faces) > 1:
                cv2.putText(frame, "More than 1 face detected!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif len(faces) == 1:
                last_face_box = faces[0]
                x, y, w, h = faces[0]['box']
                x, y = max(0, x), max(0, y)
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, IMG_SIZE)
                x_input = preprocess_input(img_to_array(face_img))
                x_input = np.expand_dims(x_input, axis=0)
                emb = model.predict(x_input)[0]
                embeddings.append(emb)

                image_path = os.path.join(person_dir, f"image_{count+1}.jpg")
                cv2.imwrite(image_path, face_img)

                count += 1
                cv2.putText(frame, f"{count}/{NUM_IMAGES}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Capture", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            
            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        if embeddings:
            save_embedding(person_id, name, embeddings)
        else:
            print("Error: No embeddings captured.")
    except Exception as e:
        print(f"Error: Failed to capture images: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

def save_embedding(person_id, name, embeddings):
    emb_mean = np.mean(embeddings, axis=0)
    emb_path = os.path.join(SAVE_DIR, 'new_vectors.npy')
    names_path = os.path.join(SAVE_DIR, 'new_names.npy')

    if os.path.exists(emb_path) and os.path.exists(names_path):
        vectors = np.load(emb_path)
        names = np.load(names_path)
        vectors = np.vstack([vectors, emb_mean])
        names = np.append(names, name)
    else:
        vectors = np.array([emb_mean])
        names = np.array([name])

    np.save(emb_path, vectors)
    np.save(names_path, names)
    print(f"Saved embedding for {name}")

    user_data = {'id': person_id, 'name': name}
    if os.path.exists(USERS_JSON_PATH):
        try:
            with open(USERS_JSON_PATH, 'r', encoding='utf-8') as f:
                users = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {USERS_JSON_PATH} is corrupted. Starting with an empty list.")
            users = []
    else:
        users = []

    users.append(user_data)
    try:
        with open(USERS_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=4)
        print(f"Saved user info to {USERS_JSON_PATH}")
    except Exception as e:
        print(f"Error: Failed to save user info to {USERS_JSON_PATH}: {e}")

if __name__ == "__main__":
    person_id = sys.argv[1]
    person_name = sys.argv[2]
    capture_images(person_id, person_name)