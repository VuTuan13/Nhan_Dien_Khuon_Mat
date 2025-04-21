import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from mtcnn import MTCNN
from model.embedding_model import load_embedding_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import csv
import os

IMG_SIZE = (224, 224)
EMB_PATH = 'embeddings/vectors.npy'
NAMES_PATH = 'embeddings/names.npy'

def recognize():
    detector = MTCNN()
    model = load_embedding_model()

    vectors = np.load(EMB_PATH)
    names = np.load(NAMES_PATH)

    cap = cv2.VideoCapture(0)
    while True:
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
            sims = cosine_similarity([emb], vectors)[0]
            max_idx = np.argmax(sims)
            if sims[max_idx] > 0.7:
                name = names[max_idx]
                mark_attendance(name)
            else:
                name = "Unknown"
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Recognize", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def mark_attendance(name):
    with open('attendance.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([name, time_str])


if __name__ == "__main__":
    recognize()
