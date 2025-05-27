import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import plot_model

# --- 1. Load model & dataset ---
best_model = tf.keras.models.load_model("model/best_resnet50.h5")

VAL_DIR = 'datasets/output/test_1'
IMG_SIZE = (224, 224)
BATCH = 32

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR, label_mode='categorical',
    image_size=IMG_SIZE, batch_size=BATCH, shuffle=False
)

class_names = val_ds.class_names
NUM_CLASSES = len(class_names)

# --- 2. Dự đoán và lấy nhãn thực ---
y_pred = []
y_true = []

for images, labels in val_ds:
    preds = best_model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

# --- 3. Ma trận nhầm lẫn ---
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
plt.figure(figsize=(12, 12))
disp.plot(cmap='Blues', xticks_rotation=90)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# --- 4. In mô hình summary ---
print("\n📋 Model Summary:")
best_model.summary()

# Nếu muốn lưu vào file txt:
with open('model_summary.txt', 'w') as f:
    best_model.summary(print_fn=lambda x: f.write(x + '\n'))

# --- 5. Vẽ sơ đồ mô hình (tạo ảnh PNG) ---
plot_model(best_model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)

# --- 6. Load lịch sử train ---
# Nếu bạn đã có biến `history_dict`, bỏ qua đoạn này
# Nếu không, bạn có thể load từ lịch sử đã lưu:
history_dict = np.load("history.npy", allow_pickle=True).item()

# --- 7. Vẽ biểu đồ Accuracy và Loss ---
plt.figure(figsize=(14,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history_dict['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history_dict['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history_dict['loss'], label='Training Loss', color='blue')
plt.plot(history_dict['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# --- 8. Bảng thống kê kết quả các epoch ---
df = pd.DataFrame(history_dict)
print("\n📊 Training Summary (Last 10 Epochs):")
print(df.tail(10))
