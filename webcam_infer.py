import cv2 as cv
import numpy as np
import mediapipe as mp
import os
import tensorflow as tf

TFLITE_PATH = "gesture_classifier.tflite"
LABELS_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.75  # Ngưỡng tin cậy (75%)

# Giảm thiểu log của TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def vector_2d_angle(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0.0

    unit_v1 = v1 / norm_v1
    unit_v2 = v2 / norm_v2

    dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot_product))

def preprocess_landmarks(landmarks_xy):
    """
    Chuẩn hóa và tính toán 36 đặc trưng hình học.
    Input: List các tọa độ [x, y] (pixel).
    Output: List 36 số thực (đã chuẩn hóa).
    """
    lm = np.array(landmarks_xy, dtype=np.float32)
    features = []

    base = lm[0].copy()      
    lm_translated = lm - base

    # Tính kích thước bàn tay để chuẩn hóa khoảng cách
    base_length = np.linalg.norm(lm_translated[12] - lm_translated[0])
    base_length = base_length if base_length > 1e-6 else 1.0

    features.extend((lm_translated[8] / base_length).tolist())

    for i in [4, 8, 12, 16, 20]:
        dist = np.linalg.norm(lm[i] - lm[0]) / base_length
        features.append(dist)
    
    pairs = [(4, 8), (4, 12), (4, 16), (4, 20), 
             (8, 12), (8, 16), (12, 16), (12, 20), (16, 20)]
    for i, j in pairs:
        dist = np.linalg.norm(lm[i] - lm[j]) / base_length
        features.append(dist)

    for i in range(5):
        base_idx = i * 4 + 1
        
        p1, p2, p3 = lm[base_idx], lm[base_idx + 1], lm[base_idx + 2]
        v1, v2 = p2 - p1, p3 - p2
        features.append(vector_2d_angle(v1, v2) / 180.0)

        p1, p2, p3 = lm[base_idx + 1], lm[base_idx + 2], lm[base_idx + 3]
        v1, v2 = p2 - p1, p3 - p2
        features.append(vector_2d_angle(v1, v2) / 180.0)

        p1, p2, p3 = lm[0], lm[base_idx], lm[base_idx + 1]
        v1, v2 = p2 - p1, p3 - p2
        features.append(vector_2d_angle(v1, v2) / 180.0)

        if i > 0:
             p1, p2, p3 = lm[base_idx], lm[base_idx+1], lm[base_idx+2]
             v1, v2 = p2 - p1, p3 - p2
             features.append(vector_2d_angle(v1, v2) / 180.0)
        else:
             features.append(0.0)

    return features


def load_labels(path):
    if not os.path.exists(path):
        print(f"Error: Label file not found at {path}")
        return []
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def draw_hand_overlay(img, hand_landmarks, img_width, img_height):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Vẽ khung xương
    mp_drawing.draw_landmarks(
        img,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 140, 255), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(50, 220, 50), thickness=2, circle_radius=2),
    )

def main():
    labels = load_labels(LABELS_PATH)
    if not labels:
        return

    if not os.path.exists(TFLITE_PATH):
        return

    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]["index"]
    out_det = interpreter.get_output_details()[0]["index"]

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    while True:
        ok, frame = cap.read()
        if not ok: break

        # Lật ảnh và lấy kích thước
        img = cv.flip(frame, 1)
        h, w = img.shape[:2]

        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        res = hands.process(rgb)

        gesture_text = ""
        status_color = (0, 0, 255) # Mặc định màu đỏ (Unknown/No hand)

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            
            # Lấy tọa độ pixel
            lm_pixel_coords = [[int(lm.x * w), int(lm.y * h)] for lm in hand.landmark]

            # Vẽ tay
            draw_hand_overlay(img, hand, w, h)

            # Tiền xử lý (tính 36 features)
            feats = preprocess_landmarks(lm_pixel_coords)
            feats_np = np.array(feats, dtype=np.float32)

            # Dự đoán bằng TFLite
            interpreter.set_tensor(in_det, np.expand_dims(feats_np, axis=0))
            interpreter.invoke()
            probs = interpreter.get_tensor(out_det)[0]

            # Lấy kết quả tốt nhất
            pred_idx = int(np.argmax(probs))
            max_prob = probs[pred_idx]

            if max_prob > CONFIDENCE_THRESHOLD:
                # Tin cậy cao -> Chấp nhận kết quả
                pred_label = labels[pred_idx]
                
                base_gesture = pred_label.rsplit("_", 1)[0] if "_" in pred_label else pred_label
                
                gesture_text = base_gesture
                status_color = (0, 255, 0) # Màu xanh lá

            else:
                # Tin cậy thấp
                gesture_text = "Unknown"
                status_color = (0, 165, 255) # Màu cam

        else:
            gesture_text = "" # Không hiện gì khi không có tay

        if gesture_text:
            (text_w, text_h), _ = cv.getTextSize(gesture_text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
            
            cv.rectangle(img, (10, 10), (10 + text_w + 20, 10 + text_h + 40), (0, 0, 0), -1)
            
            cv.putText(img, gesture_text, (20, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv.imshow("Gesture Recognition (Display Only)", img)
        
        # Nhấn Esc để thoát
        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()