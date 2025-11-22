import csv
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Cấu hình ---
CSV_PATH = "keypoints.csv"
H5_PATH = "gesture.h5"
TFLITE_PATH = "gesture.tflite"
LABELS_PATH = "labels_1.txt"

def load_dataset(csv_path):
    X = []
    y_labels = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Bỏ qua header
        for row in reader:
            if not row: continue
            # Cột 0 là label, các cột sau là features
            y_labels.append(row[0])
            # Chuyển dữ liệu features sang float
            try:
                X.append([float(x) for x in row[1:]])
            except ValueError:
                continue
    return np.array(X, dtype=np.float32), y_labels

def main():
    # 1. Load dữ liệu
    if not os.path.exists(CSV_PATH):
        print(f"Lỗi: Không tìm thấy file {CSV_PATH}. Hãy chạy build_dataset.py trước.")
        return

    X, labels = load_dataset(CSV_PATH)
    if len(X) == 0:
        print("Dữ liệu rỗng.")
        return

    print(f"Input shape: {X.shape}") # Nên là (số mẫu, 42)

    # 2. Mã hóa nhãn (Label Encoding)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    num_classes = len(le.classes_)
    print(f"Số lớp (classes): {num_classes} -> {le.classes_}")

    # 3. Chia tập train/test
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Xây dựng Model MLP đơn giản (Input = 42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(42,)),  # 21 điểm * 2 (x, y)
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 5. Train
    print("Bắt đầu training...")
    # Callback dừng sớm nếu không cải thiện
    es = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[es]
    )

    # 6. Đánh giá
    loss, acc = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {acc:.4f}")

    # 7. Lưu Model & Labels
    model.save(H5_PATH)
    
    with open(LABELS_PATH, "w") as f:
        for label in le.classes_:
            f.write(label + "\n")

    # 8. Convert sang TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    print("Hoàn tất! Đã lưu model và labels.")

if __name__ == "__main__":
    main()