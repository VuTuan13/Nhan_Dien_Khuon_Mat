import os
import shutil
from sklearn.model_selection import train_test_split

# Đường dẫn đến thư mục chứa 50 người đã được lọc
source_dir = '../datasets/output/selected_actors_100'
output_train = '../datasets/output/train_1'
output_test = '../datasets/output/test_1'

# Tạo thư mục đầu ra nếu chưa có
os.makedirs(output_train, exist_ok=True)
os.makedirs(output_test, exist_ok=True)

# Duyệt qua từng người
for person_name in os.listdir(source_dir):
    person_path = os.path.join(source_dir, person_name)
    if os.path.isdir(person_path):
        # Lấy tất cả ảnh trong thư mục của người này
        images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Tạo đường dẫn đầy đủ cho ảnh
        image_paths = [os.path.join(person_path, img) for img in images]

        # Chia 80% train, 20% test
        train_paths, test_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

        # Tạo thư mục cho người đó trong train/test
        train_person_dir = os.path.join(output_train, person_name)
        test_person_dir = os.path.join(output_test, person_name)
        os.makedirs(train_person_dir, exist_ok=True)
        os.makedirs(test_person_dir, exist_ok=True)

        # Copy ảnh vào tập train
        for src in train_paths:
            dst = os.path.join(train_person_dir, os.path.basename(src))
            shutil.copy2(src, dst)

        # Copy ảnh vào tập test
        for src in test_paths:
            dst = os.path.join(test_person_dir, os.path.basename(src))
            shutil.copy2(src, dst)

print("🎉 Đã chia thành công tập train/test với tỉ lệ 80/20 cho tất cả người.")
