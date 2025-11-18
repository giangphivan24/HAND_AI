# # webcam_infer.py
# import cv2 as cv
# import numpy as np
# import mediapipe as mp
# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # silence oneDNN variance if desired
# import tensorflow as tf

# TFLITE_PATH = "gesture_classifier.tflite"
# LABELS_PATH = "labels.txt"

# def load_labels(path):
#     with open(path, "r") as f:
#         return [line.strip() for line in f if line.strip()]

# def preprocess_landmarks(landmarks_xy):
#     # Match the exact preprocessing from build_dataset.py for consistency
#     lm = np.array(landmarks_xy, dtype=np.float32)
#     base = lm[0].copy()  # Lấy cổ tay làm gốc tọa độ
#     lm -= base  # Dời toàn bộ điểm về gốc
#     max_val = np.max(np.abs(lm)) if np.max(np.abs(lm)) > 0 else 1.0  # Tìm giá trị lớn nhất để scale
#     lm = lm / max_val  # Scale về [-1, 1]
#     xs = lm[:, 0]
#     ys = lm[:, 1]
#     # Return as numpy array (same format, but as array for TensorFlow compatibility)
#     return np.concatenate([xs, ys]).astype(np.float32)


# def draw_hand_overlay(img, hand_landmarks, img_width, img_height):
#     mp_drawing = mp.solutions.drawing_utils
#     mp_hands = mp.solutions.hands

#     # Draw skeleton connections
#     mp_drawing.draw_landmarks(
#         img,
#         hand_landmarks,
#         mp_hands.HAND_CONNECTIONS,
#         mp_drawing.DrawingSpec(color=(0, 140, 255), thickness=2, circle_radius=3),
#         mp_drawing.DrawingSpec(color=(50, 220, 50), thickness=2, circle_radius=2),
#     )

#     # Build list of pixel points for convex hull
#     pts = []
#     for lm in hand_landmarks.landmark:
#         pts.append([int(lm.x * img_width), int(lm.y * img_height)])
#     if len(pts) >= 3:
#         pts = np.array(pts, dtype=np.int32)
#         hull = cv.convexHull(pts)

#         # Translucent fill for hand outline
#         overlay = img.copy()
#         cv.fillConvexPoly(overlay, hull, (0, 255, 0))
#         cv.addWeighted(overlay, 0.18, img, 0.82, 0, img)

# def main():
#     labels = load_labels(LABELS_PATH)
#     interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
#     interpreter.allocate_tensors()
#     in_det = interpreter.get_input_details()[0]["index"]
#     out_det = interpreter.get_output_details()[0]["index"]

#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
#                            min_detection_confidence=0.6, min_tracking_confidence=0.5)

#     cap = cv.VideoCapture(0)
#     if not cap.isOpened():
#         print("Cannot open camera")
#         return

#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break

#         img = cv.flip(frame, 1)
#         rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#         res = hands.process(rgb)

#         gesture_text = "No hand"
#         if res.multi_hand_landmarks:
#             hand = res.multi_hand_landmarks[0]
#             # Determine handedness using MediaPipe instead of the classifier
#             side_text = None
#             if getattr(res, "multi_handedness", None):
#                 try:
#                     side_text = res.multi_handedness[0].classification[0].label  # 'Left' or 'Right'
#                 except Exception:
#                     side_text = None
#             h, w = img.shape[:2]
#             landmarks = []
#             for lm in hand.landmark:
#                 landmarks.append([int(lm.x * w), int(lm.y * h)])

#             # Draw enhanced overlay (skeleton + translucent hull)
#             draw_hand_overlay(img, hand, w, h)

#             feats = preprocess_landmarks(landmarks)

#             interpreter.set_tensor(in_det, np.expand_dims(feats, axis=0))
#             interpreter.invoke()
#             probs = interpreter.get_tensor(out_det)[0]
#             pred_idx = int(np.argmax(probs))
#             pred_label = labels[pred_idx]
#             # If labels are like 'Like_Left', replace the side with MediaPipe's side
#             if "_" in pred_label and side_text is not None:
#                 base_gesture = pred_label.rsplit("_", 1)[0]
#                 gesture_text = f"{base_gesture}_{side_text} ({probs[pred_idx]:.2f})"
#             else:
#                 # If no side in label list, optionally append detected side
#                 if side_text is not None:
#                     gesture_text = f"{pred_label}_{side_text} ({probs[pred_idx]:.2f})"
#                 else:
#                     gesture_text = f"{pred_label} ({probs[pred_idx]:.2f})"

#             # Optionally draw small points on top for precision
#             for x, y in landmarks:
#                 cv.circle(img, (x, y), 2, (0, 120, 0), -1)

#         cv.putText(img, gesture_text, (10, 40),
#                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
#         cv.putText(img, gesture_text, (10, 40),
#                    cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#         cv.imshow("Palm Gesture Recognition", img)
#         if cv.waitKey(1) & 0xFF == 27:  # ESC to quit
#             break

#     cap.release()
#     cv.destroyAllWindows()
#     hands.close()

# if __name__ == "__main__":
#     main()
    
    



# webcam_infer.py (Phiên bản Cải tiến)
import cv2 as cv
import numpy as np
import mediapipe as mp
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Giảm log verbosity của TF
import tensorflow as tf
import pyautogui # Thư viện điều khiển chuột/phím

# --- Cấu hình ---
TFLITE_PATH = "gesture_classifier.tflite"
LABELS_PATH = "labels.txt"
# --- End Cấu hình ---


# --- HÀM HỖ TRỢ ĐIỀU KHIỂN HỆ THỐNG ---
# Khai báo các trạng thái cử chỉ
last_gesture = "No hand"
is_controlling_mouse = False
mouse_control_lm_index = 8 # Dùng đầu ngón trỏ (8) để di chuyển chuột

def execute_action(gesture, lm_coords, w, h):
    global last_gesture, is_controlling_mouse

    current_gesture = gesture.split('(')[0].strip().rsplit("_", 1)[0] # Lấy nhãn chính (vd: '1', '0', 'Like')

    # Chuyển đổi tọa độ landmark (pixel) sang tọa độ màn hình (screen)
    # Tọa độ màn hình (0,0) là góc trên bên trái
    screen_width, screen_height = pyautogui.size()
    
    # Lấy tọa độ điểm mốc (LM 8 - đầu ngón trỏ)
    if len(lm_coords) > mouse_control_lm_index:
        x_cam, y_cam = lm_coords[mouse_control_lm_index] # Tọa độ pixel trong camera
        
        # Chuyển đổi tọa độ
        # Tọa độ x_screen sẽ tỷ lệ thuận với x_cam
        # Tọa độ y_screen sẽ tỷ lệ thuận với y_cam
        x_screen = int(np.interp(x_cam, [0, w], [0, screen_width]))
        y_screen = int(np.interp(y_cam, [0, h], [0, screen_height]))
    else:
        x_screen, y_screen = None, None

    # --- 1. Xử lý Trạng thái ---
    
    # Hành động chụp màn hình (chỉ thực hiện khi cử chỉ VỪA thay đổi sang '0')
    if current_gesture == '0' and last_gesture != '0':
        print("Action: Screenshot (Ctrl+PrintScreen)")
        pyautogui.hotkey('ctrl', 'printscreen')
        is_controlling_mouse = False

    # Chuyển đổi chế độ điều khiển chuột
    if current_gesture == '1': # Bật chế độ điều khiển chuột
        is_controlling_mouse = True
    elif current_gesture == '5' and last_gesture != '5': # Tắt chế độ điều khiển chuột (ví dụ: xòe 5 ngón)
        is_controlling_mouse = False

    # --- 2. Điều khiển Chuột (Nếu đang ở chế độ '1') ---
    
    if is_controlling_mouse and x_screen is not None:
        # Di chuyển chuột theo ngón trỏ (LM 8)
        pyautogui.moveTo(x_screen, y_screen, duration=0.0)

    # Click chuột (Ví dụ: Cử chỉ '2' - hai ngón)
    if current_gesture == '2' and last_gesture != '2':
        print("Action: Double Click")
        pyautogui.doubleClick(button='left')
        
    # --- 3. Cập nhật trạng thái cuối cùng ---
    last_gesture = current_gesture


# --- HÀM XỬ LÝ ĐẶC TRƯNG CẢI TIẾN (Đảm bảo giống hệt build_dataset.py) ---

def vector_2d_angle(v1, v2):
    """Tính góc giữa hai vector 2D v1 và v2 (tính bằng độ)."""
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    
    unit_v1 = v1 / norm_v1
    unit_v2 = v2 / norm_v2
    
    dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot_product))
    return angle

def preprocess_landmarks(landmarks_xy):
    """
    Chuẩn hóa và tính toán 36 đặc trưng hình học (khoảng cách và góc).
    Sử dụng 21 điểm mốc (chỉ x, y).
    """
    lm = np.array(landmarks_xy, dtype=np.float32)[:, :2] 
    features = []
    
    base = lm[0].copy()
    lm_translated = lm - base
    
    base_length = np.linalg.norm(lm_translated[12] - lm_translated[0])
    base_length = base_length if base_length > 1e-6 else 1.0 
    
    # 3.1. [2 features]: Tọa độ đã chuẩn hóa của một điểm tham chiếu
    features.extend((lm_translated[8] / base_length).tolist()) 

    # 3.2. [14 features]: Khoảng cách đã chuẩn hóa
    for i in [4, 8, 12, 16, 20]:
        dist = np.linalg.norm(lm[i] - lm[0]) / base_length
        features.append(dist)
        
    for i, j in [(4, 8), (4, 12), (4, 16), (4, 20), (8, 12), (8, 16), (12, 16), (12, 20), (16, 20)]:
        dist = np.linalg.norm(lm[i] - lm[j]) / base_length
        features.append(dist)
        
    # 3.3. [20 features]: Góc bất biến xoay
    for i in range(5):
        base_idx = i * 4 + 1
        
        # Góc 1: Nút (X) - (X+1) - (X+2)
        p1, p2, p3 = lm[base_idx], lm[base_idx + 1], lm[base_idx + 2]
        v1, v2 = p2 - p1, p3 - p2
        features.append(vector_2d_angle(v1, v2) / 180.0) 

        # Góc 2: Nút (X+1) - (X+2) - (X+3)
        p1, p2, p3 = lm[base_idx + 1], lm[base_idx + 2], lm[base_idx + 3]
        v1, v2 = p2 - p1, p3 - p2
        features.append(vector_2d_angle(v1, v2) / 180.0)
        
        # Góc 3: Góc giữa vector Cổ tay - Gốc ngón (0-X) và Đốt thứ nhất (X-X+1)
        p1, p2, p3 = lm[0], lm[base_idx], lm[base_idx + 1]
        v1, v2 = p2 - p1, p3 - p2
        features.append(vector_2d_angle(v1, v2) / 180.0)
        
    return np.array(features, dtype=np.float32)

# --- HÀM MAIN ---

def load_labels(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def draw_hand_overlay(img, hand_landmarks, img_width, img_height):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Vẽ khung xương và kết nối
    mp_drawing.draw_landmarks(
        img,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 140, 255), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(50, 220, 50), thickness=2, circle_radius=2),
    )

    # Vẽ khung bao (Convex Hull) trong suốt
    pts = []
    for lm in hand_landmarks.landmark:
        pts.append([int(lm.x * img_width), int(lm.y * img_height)])
    if len(pts) >= 3:
        pts = np.array(pts, dtype=np.int32)
        hull = cv.convexHull(pts)

        overlay = img.copy()
        cv.fillConvexPoly(overlay, hull, (0, 255, 0))
        cv.addWeighted(overlay, 0.18, img, 0.82, 0, img)

def main():
    labels = load_labels(LABELS_PATH)
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]["index"]
    out_det = interpreter.get_output_details()[0]["index"]

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.6, min_tracking_confidence=0.5)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Kích thước features đầu vào phải là 36
    EXPECTED_FEATURE_LENGTH = 36
    if interpreter.get_input_details()[0]['shape'][1] != EXPECTED_FEATURE_LENGTH:
        print(f"ERROR: Model input size is {interpreter.get_input_details()[0]['shape'][1]}, but expected {EXPECTED_FEATURE_LENGTH} features.")
        return

    while True:
        ok, frame = cap.read()
        if not ok: break

        img = cv.flip(frame, 1)
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        res = hands.process(rgb)

        gesture_text = "No hand"
        lm_pixel_coords = []
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            
            side_text = res.multi_handedness[0].classification[0].label if getattr(res, "multi_handedness", None) else None
            
            h, w = img.shape[:2]
            
            # Lấy tọa độ pixel của 21 điểm mốc
            lm_pixel_coords = [[int(lm.x * w), int(lm.y * h)] for lm in hand.landmark]

            # Vẽ overlay
            draw_hand_overlay(img, hand, w, h)

            # TIỀN XỬ LÝ ĐẶC TRƯNG
            feats = preprocess_landmarks(lm_pixel_coords)

            # DỰ ĐOÁN
            interpreter.set_tensor(in_det, np.expand_dims(feats, axis=0))
            interpreter.invoke()
            probs = interpreter.get_tensor(out_det)[0]
            pred_idx = int(np.argmax(probs))
            pred_label = labels[pred_idx]
            
            # Hiển thị kết quả
            base_gesture = pred_label.rsplit("_", 1)[0] if "_" in pred_label else pred_label
            display_text = f"{base_gesture} ({probs[pred_idx]:.2f})"
            if side_text:
                 display_text += f" - {side_text}"
            
            gesture_text = display_text
            
            # THỰC HIỆN HÀNH ĐỘNG
            execute_action(gesture_text, lm_pixel_coords, w, h)


        cv.putText(img, gesture_text, (10, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv.putText(img, gesture_text, (10, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Hiển thị trạng thái điều khiển
        control_status = "Mouse Control: ON" if is_controlling_mouse else "Mouse Control: OFF"
        color = (0, 255, 0) if is_controlling_mouse else (0, 0, 255)
        cv.putText(img, control_status, (10, h - 20), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


        cv.imshow("Palm Gesture Recognition", img)
        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()