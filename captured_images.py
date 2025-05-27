import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from mtcnn import MTCNN
from model.embedding_model import load_embedding_model
from PIL import Image

# Cấu hình thư mục
TRAIN_DIR = 'datasets/output/train_1'
TEST_DIR = 'datasets/output/test_1'
IMG_SIZE = (224, 224)
EMB_PATH = 'embeddings/vectors.npy'
NAMES_PATH = 'embeddings/names.npy'

def create_embeddings():
    detector = MTCNN()
    model = load_embedding_model()

    vectors = []
    names = []

    def process_images_from_directory(directory):
        nonlocal vectors, names
        total = 0

        for person_dir in os.listdir(directory):
            person_path = os.path.join(directory, person_dir)
            if not os.path.isdir(person_path):
                continue

            # ✅ Giữ nguyên tên thư mục làm tên người
            name = person_dir

            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)

                try:
                    image = Image.open(image_path).convert("RGB")
                    image = np.array(image)
                except Exception as e:
                    print(f"[!] Không thể đọc ảnh {image_path}: {e}")
                    continue

                faces = detector.detect_faces(image)
                if len(faces) == 1:
                    x, y, w, h = faces[0]['box']
                    x, y = max(0, x), max(0, y)
                    face_img = image[y:y+h, x:x+w]

                    if face_img.shape[0] == 0 or face_img.shape[1] == 0:
                        print(f"[!] Vùng khuôn mặt lỗi trong ảnh {image_path}")
                        continue

                    face_img = cv2.resize(face_img, IMG_SIZE)
                    x_input = preprocess_input(img_to_array(face_img))
                    x_input = np.expand_dims(x_input, axis=0)

                    emb = model.predict(x_input)[0]
                    vectors.append(emb)
                    names.append(name)
                    total += 1
                else:
                    print(f"[!] Ảnh {image_path} có {len(faces)} khuôn mặt. Bỏ qua.")

        print(f"✅ Đã xử lý {total} ảnh từ thư mục: {directory}")

    # Chạy cho cả train và test
    process_images_from_directory(TRAIN_DIR)
    process_images_from_directory(TEST_DIR)

    vectors = np.array(vectors)
    names = np.array(names)

    # Lưu kết quả
    os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
    np.save(EMB_PATH, vectors)
    np.save(NAMES_PATH, names)
    print(f"\n✅ Đã lưu embeddings vào:\n➡️ {EMB_PATH}\n➡️ {NAMES_PATH}")

# Chạy
if __name__ == '__main__':
    create_embeddings()
