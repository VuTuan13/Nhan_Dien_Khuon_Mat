from flask import Flask, render_template, request, redirect, url_for, flash
import subprocess
import numpy as np
import cv2
from PIL import Image
from mtcnn import MTCNN
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from model.embedding_model import load_embedding_model
import csv
from datetime import datetime
import os
import pandas as pd
import json

app = Flask(__name__)
app.secret_key = 'supersecretkey'

detector = MTCNN()
embedding_model = load_embedding_model()
known_vectors = np.load('embeddings/vectors.npy')
known_names = np.load('embeddings/names.npy')
LOG_FILE = 'attendance_log.csv'
USERS_JSON_PATH = 'embeddings/users.json'

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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

def save_attendance_log(name, status):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_id = get_user_id(name)
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([user_id, name, now, status])

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/history')
def show_history():
    try:
        with open('history.csv', mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            records = []
            for row in reader:
                if len(row) == 4:  # Đọc 4 cột: id, name, time, status
                    user_id, name, time, status = row
                    records.append({'id': user_id, 'name': name, 'time': time, 'status': status})
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        records = []
    return render_template('history.html', records=records)

@app.route('/add', methods=['POST'])
def add_person():
    person_id = request.form.get('id')
    name = request.form.get('name')
    result = subprocess.run(["python", "scripts/new_person.py", person_id, name], capture_output=True, text=True)
    if "Error" in result.stdout:
        flash("Error: Person with this ID already exists.", 'error')
    else:
        flash(f"Successfully added person {name} with ID {person_id}.", 'success')
    return redirect(url_for('index'))

@app.route('/check_in', methods=['GET'])
def check_in():
    subprocess.run(["python", "scripts/recognize.py", "Check-in"])
    return redirect("/")

@app.route('/check_out', methods=['GET'])
def check_out():
    subprocess.run(["python", "scripts/recognize.py", "Check-out"])
    return redirect("/")

@app.route('/recognize_image', methods=['POST'])
def recognize_image():
    file = request.files['image']
    if not file:
        flash("❌ Không có ảnh được chọn", "error")
        return redirect('/')

    try:
        image = Image.open(file).convert('RGB')
        image = np.array(image)
        faces = detector.detect_faces(image)

        if len(faces) != 1:
            flash(f"❌ Ảnh có {len(faces)} khuôn mặt, cần đúng 1 khuôn mặt.", "error")
            return redirect('/')

        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)
        face_img = image[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))

        x_input = preprocess_input(img_to_array(face_img))
        x_input = np.expand_dims(x_input, axis=0)
        emb = embedding_model.predict(x_input)[0]

        similarities = [cosine_similarity(emb, vec) for vec in known_vectors]
        max_index = np.argmax(similarities)
        best_score = similarities[max_index]

        if best_score > 0.5:
            name = known_names[max_index]
            save_attendance_log(name, "Check-in")  # Mặc định là Check-in cho ảnh
            flash(f"✅ Khuôn mặt được nhận dạng là: {name} (score={best_score:.2f})", "success")
        else:
            flash("❌ Không nhận dạng được khuôn mặt trong dữ liệu", "error")

    except Exception as e:
        flash(f"❌ Lỗi khi xử lý ảnh: {e}", "error")

    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)