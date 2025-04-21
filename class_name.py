import os
import json

DATASET_DIR = '../datasets/output/selected_actors_100'  
class_names = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

with open('../scripts/class_names.json', 'w') as f:
    json.dump(class_names, f)

print("Đã lưu class_names.json với", len(class_names), "nhãn.")
