import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import os
from mtcnn import MTCNN
from model.embedding_model import load_embedding_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

SAVE_DIR = 'embeddings'
NUM_IMAGES = 100
IMG_SIZE = (224, 224)

def capture_images(name):
    detector = MTCNN()
    model = load_embedding_model()
    embeddings = []

    cap = cv2.VideoCapture(0)
    count = 0
    print(f"Capturing images for: {name}")

    while count < NUM_IMAGES:
        ret, frame = cap.read()
        if not ret:
            continue
        faces = detector.detect_faces(frame)
        for face in faces:
            x, y, w, h = face['box']
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, IMG_SIZE)
            x_input = preprocess_input(img_to_array(face_img))
            x_input = np.expand_dims(x_input, axis=0)
            emb = model.predict(x_input)[0]
            embeddings.append(emb)
            count += 1
            cv2.putText(frame, f"{count}/{NUM_IMAGES}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) > 0:
        save_embedding(name, embeddings)

def save_embedding(name, embeddings):
    emb_mean = np.mean(embeddings, axis=0)
    emb_path = os.path.join(SAVE_DIR, 'vectors.npy')
    names_path = os.path.join(SAVE_DIR, 'names.npy')

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

if __name__ == "__main__":
    person_name = input("Enter person's name: ")
    capture_images(person_name)
