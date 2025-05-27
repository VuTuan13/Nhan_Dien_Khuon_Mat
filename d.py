import numpy as np

# Đường dẫn tới 2 file embeddings
names_path = 'embeddings/names.npy'

# Đọc file
names = np.load(names_path)

# In tên và embedding tương ứng
for i, name in enumerate(names):
    print(f"🧑‍🦱 Tên: {name}")
    print("-" * 50)

# Thống kê
print(f"\n📊 Tổng số embeddings: {len(names)}")

