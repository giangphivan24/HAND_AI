# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# import csv
# import numpy as np
# import cv2 as cv
# import mediapipe as mp
# from absl import logging as absl_logging

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(SCRIPT_DIR, "TEST1")
# OUT_CSV = "keypoints_dataset.csv"

# # --- 1. HÀM TIỀN XỬ LÝ CHO PHƯƠNG PHÁP 42 ĐIỂM (RAW COORDS) ---
# def preprocess_landmarks(landmarks_xy):
#     """
#     Giữ nguyên logic cũ mà bạn thấy hiệu quả:
#     - Dời về gốc (0,0) tại cổ tay.
#     - Flatten (trải phẳng) thành mảng 1 chiều.
#     - Chia cho giá trị tuyệt đối lớn nhất để chuẩn hóa về [-1, 1].
#     Output: 42 số thực.
#     """
#     lm = np.array(landmarks_xy, dtype=np.float32)
#     base = lm[0].copy()  # Lấy cổ tay làm gốc
#     lm -= base           # Dời hình về gốc
#     lm = lm.flatten()    # Trải phẳng: [x0, y0, x1, y1, ...]
    
#     # Tìm giá trị lớn nhất để scale
#     max_val = np.max(np.abs(lm))
#     if max_val > 0:
#         lm = lm / max_val
        
#     return lm.tolist()

# # --- 2. HÀM ĐỌC ẢNH (HỖ TRỢ TIẾNG VIỆT/UNICODE) ---
# def imread_unicode(path):
#     try:
#         # Đọc file binary rồi decode để tránh lỗi đường dẫn tiếng Việt trên Windows
#         stream = open(path, "rb")
#         bytes = bytearray(stream.read())
#         numpyarray = np.asarray(bytes, dtype=np.uint8)
#         return cv.imdecode(numpyarray, cv.IMREAD_COLOR)
#     except Exception:
#         return None

# # --- 3. HÀM DUYỆT ẢNH (CẬP NHẬT CHO CẤU TRÚC MỚI) ---
# def iter_images(root_dir):
#     """
#     Duyệt cấu trúc:
#     TEST1/ (root)
#       ├── 0/ (class_name -> LABEL)
#       │    ├── 0.1/ (session - bỏ qua tên này)
#       │    │    ├── Left/ (side - chỉ để duyệt, không dùng làm label chính)
#       │    │    │    └── IMG...
#       │    │    └── Right/
#       │    └── 0.2/ ...
#       ├── 1/ ...
#     """
#     # Tầng 1: Tên nhãn chính (0, 1, ...)
#     for class_name in sorted(os.listdir(root_dir)):
#         class_dir = os.path.join(root_dir, class_name)
#         if not os.path.isdir(class_dir): continue

#         # Tầng 2: Các folder con (0.1, 0.2, 1.1...) - Chỉ dùng để duyệt tiếp
#         for session_name in sorted(os.listdir(class_dir)):
#             session_dir = os.path.join(class_dir, session_name)
#             if not os.path.isdir(session_dir): continue

#             # Tầng 3: Left / Right
#             for side_name in sorted(os.listdir(session_dir)):
#                 side_dir = os.path.join(session_dir, side_name)
#                 if not os.path.isdir(side_dir): continue

#                 # Quyết định nhãn: Dùng class_name ("0", "1") làm nhãn.
#                 # (Mô hình sẽ tự học rằng label "0" có 2 kiểu hình dáng là trái và phải)
#                 current_label = class_name

#                 # Tầng 4: File ảnh
#                 for fname in os.listdir(side_dir):
#                     if fname.lower().endswith((".jpg", ".jpeg", ".png")):
#                         yield current_label, os.path.join(side_dir, fname)

# def main():
#     mp_hands = mp.solutions.hands
#     # Lưu ý: max_num_hands=1 để tập trung vào tay chính trong ảnh training
#     hands = mp_hands.Hands(
#         static_image_mode=True,
#         max_num_hands=1, 
#         model_complexity=1,
#         min_detection_confidence=0.3,
#     )

#     if not os.path.isdir(DATA_DIR):
#         print(f"Lỗi: Không tìm thấy thư mục {DATA_DIR}")
#         return

#     print(f"Đang quét dữ liệu từ: {DATA_DIR}")
    
#     # Mở file CSV để ghi
#     with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
        
#         # Header cho 42 điểm (x0, y0 ... x20, y20)
#         header = ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
#         writer.writerow(header)

#         count = 0
#         skipped = 0
        
#         for label, path in iter_images(DATA_DIR):
#             img = imread_unicode(path)
#             if img is None:
#                 skipped += 1
#                 continue

#             rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#             res = hands.process(rgb)

#             if res.multi_hand_landmarks:
#                 # Lấy bàn tay đầu tiên phát hiện được
#                 hand = res.multi_hand_landmarks[0]
#                 h, w = img.shape[:2]
                
#                 # Lấy tọa độ pixel
#                 landmarks_pixel = [[int(lm.x * w), int(lm.y * h)] for lm in hand.landmark]

#                 # Tiền xử lý (42 điểm)
#                 features = preprocess_landmarks(landmarks_pixel)
                
#                 # Ghi vào CSV
#                 writer.writerow([label] + features)
#                 count += 1
#                 if count % 100 == 0:
#                     print(f"Đã xử lý {count} ảnh (Label hiện tại: {label})...")
#             else:
#                 skipped += 1

#     hands.close()
#     print(f"--- HOÀN TẤT ---")
#     print(f"Đã lưu: {count} mẫu.")
#     print(f"Bỏ qua: {skipped} ảnh (lỗi đọc hoặc không thấy tay).")
#     print(f"File output: {OUT_CSV}")

# if __name__ == "__main__":
#     absl_logging.set_verbosity(absl_logging.ERROR)
#     main()



import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import csv
import numpy as np
import cv2 as cv
import mediapipe as mp
from absl import logging as absl_logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "TEST1")
OUT_CSV = "keypoints.csv"

# --- HÀM TIỀN XỬ LÝ (42 ĐIỂM) ---
def preprocess_landmarks(landmarks_xy):
    lm = np.array(landmarks_xy, dtype=np.float32)
    base = lm[0].copy()
    lm -= base
    lm = lm.flatten()
    max_val = np.max(np.abs(lm))
    if max_val > 0:
        lm = lm / max_val
    return lm.tolist()

# --- HÀM ĐỌC ẢNH ---
def imread_unicode(path):
    try:
        stream = open(path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        return cv.imdecode(numpyarray, cv.IMREAD_COLOR)
    except Exception:
        return None

# --- HÀM DUYỆT ẢNH (SỬA LOGIC NHÃN TẠI ĐÂY) ---
def iter_images(root_dir):
    # Cấu trúc: TEST1 / 0 / 0.1 / Left / img.jpg
    for class_name in sorted(os.listdir(root_dir)): # class_name: "0", "1"
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir): continue

        for session_name in sorted(os.listdir(class_dir)): # session_name: "0.1"
            session_dir = os.path.join(class_dir, session_name)
            if not os.path.isdir(session_dir): continue

            for side_name in sorted(os.listdir(session_dir)): # side_name: "Left", "Right"
                side_dir = os.path.join(session_dir, side_name)
                if not os.path.isdir(side_dir): continue

                # --- THAY ĐỔI QUAN TRỌNG Ở ĐÂY ---
                # Ghép tên Class và tên Side thành nhãn duy nhất
                # Ví dụ: "0" + "_" + "Left" = "0_Left"
                label_full = f"{class_name}_{side_name}" 

                for fname in os.listdir(side_dir):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        yield label_full, os.path.join(side_dir, fname)

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1, 
        model_complexity=1,
        min_detection_confidence=0.3,
    )

    if not os.path.isdir(DATA_DIR):
        print(f"Lỗi: Không tìm thấy {DATA_DIR}")
        return

    print(f"Đang quét dữ liệu...")
    
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
        writer.writerow(header)

        count = 0
        for label, path in iter_images(DATA_DIR):
            img = imread_unicode(path)
            if img is None: continue

            rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                h, w = img.shape[:2]
                landmarks_pixel = [[int(lm.x * w), int(lm.y * h)] for lm in hand.landmark]
                features = preprocess_landmarks(landmarks_pixel)
                
                writer.writerow([label] + features)
                count += 1
                if count % 100 == 0: print(f"Đã xử lý {count} ảnh ({label})...")

    hands.close()
    print(f"Hoàn tất! Đã lưu {count} mẫu vào {OUT_CSV}")

if __name__ == "__main__":
    absl_logging.set_verbosity(absl_logging.ERROR)
    main()