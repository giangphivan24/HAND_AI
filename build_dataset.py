import os
# Đặt biến môi trường để tắt các thông báo log "Info" và "Warning" của TensorFlow
# Chỉ hiển thị lỗi (Error) nếu có.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import csv # Thư viện để đọc/ghi file CSV
import numpy as np # Thư viện xử lý mảng số
import cv2 as cv # Thư viện xử lý ảnh OpenCV
import mediapipe as mp # Thư viện MediaPipe để phát hiện bàn tay
from absl import logging as absl_logging # Thư viện để quản lý log của MediaPipe


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Xác định đường dẫn thư mục hiện tại
DATA_DIR = os.path.join(SCRIPT_DIR, "AI") # Thư mục chứa dữ liệu ảnh 
OUT_CSV = "keypoints_dataset.csv" 


# Tính góc giữa hai vector 2D v1 và v2
def vector_2d_angle(v1, v2):
    norm_v1 = np.linalg.norm(v1) # Tính độ dài (magnitude/norm) của vector v1
    norm_v2 = np.linalg.norm(v2) # Tính độ dài của vector v2
    
    # Nếu một trong hai vector có độ dài bằng 0, trả về góc 0 
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    
    # Chuẩn hóa hai vector về dạng đơn vị (unit vector)
    # Công thức: u = v / ||v||
    unit_v1 = v1 / norm_v1
    unit_v2 = v2 / norm_v2
    
    # Tính tích vô hướng (dot product) giữa hai vector đơn vị
    # np.clip để giới hạn giá trị trong khoảng [-1, 1]
    # Công thức: dot(u1, u2) = ||u1|| * ||u2|| * cos(theta) = cos(theta)
    dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    
    # Tính góc (theta) giữa hai vector bằng cách lấy arccos của tích vô hướng
    return np.degrees(np.arccos(dot_product))


# Chuẩn hóa và tính toán 36 đặc trưng hình học
def preprocess_landmarks(landmarks_xy):
    """
    Hàm này nhận vào danh sách tọa độ (x, y) của 21 khớp tay.
    Trả về danh sách 36 đặc trưng hình học đã được chuẩn hóa.
    """
    # Chuyển danh sách tọa độ sang mảng numpy
    lm = np.array(landmarks_xy, dtype=np.float32)
    features = []
   
    # Lấy tọa độ cổ tay làm gốc (điểm 0) 
    base = lm[0].copy()
    
    # Dời tất cả các điểm sao cho Cổ tay về tọa độ (0,0)
    lm_translated = lm - base
    
    # Khoảng cách từ cổ tay đến đầu ngón giữa
    base_length = np.linalg.norm(lm_translated[12] - lm_translated[0]) 
    base_length = base_length if base_length > 1e-6 else 1.0 
    
    # Tọa độ đầu ngón trỏ (điểm 8)
    # Nhận biết bàn tay đang hướng về đâu (lên/xuống/trái/phải)
    features.extend((lm_translated[8] / base_length).tolist()) 

    # Khoảng cách từ cổ tay đến 5 đầu ngón tay
    # Nhận biết co/duỗi ngón tay
    for i in [4, 8, 12, 16, 20]:
        dist = np.linalg.norm(lm[i] - lm[0]) / base_length
        features.append(dist)
        
    # Khoảng cách giữa các đầu ngón tay với nhau và các khớp
    # Nhận biết đang chụm lại hay xòe ra
    for i, j in [(4, 8), (4, 12), (4, 16), (4, 20), (8, 12), (8, 16), (12, 16), (12, 20), (16, 20)]:
        dist = np.linalg.norm(lm[i] - lm[j]) / base_length
        features.append(dist)
        
    # Góc gập ngón tay
    # Duyệt qua 5 ngón tay
    for i in range(5):
        # Tính chỉ số bắt đầu của ngón đó trong danh sách 21 điểm
        # Công thức: i*4 + 1 (ngón cái: 1, ngón trỏ: 5, ngón giữa: 9, ngón áp út: 13, ngón út: 17)
        base_idx = i * 4 + 1
        
        # Tính 3 góc quan trọng trên mỗi ngón
        # p1: khớp gốc, p2: khớp giữa, p3: khớp đầu
        
        # Góc 1: Độ gập của khớp giữa
        # Cho biết ngón tay đang gập hay duỗi
        p1, p2, p3 = lm[base_idx], lm[base_idx + 1], lm[base_idx + 2] 
        v1, v2 = p2 - p1, p3 - p2
        features.append(vector_2d_angle(v1, v2) / 180.0) 
        
        # Góc 2: Độ gập của khớp đầu (góc giữa xương đốt 2 và 3)
        # cho biết đầu ngón tay có bị quặp xuống hay không
        p1, p2, p3 = lm[base_idx + 1], lm[base_idx + 2], lm[base_idx + 3]
        v1, v2 = p2 - p1, p3 - p2
        features.append(vector_2d_angle(v1, v2) / 180.0)
        
        # Góc 3: góc giữa các ngón tay và lòng bàn tay
        # cho biết ngón tay đang dựng lên hay cụp xuống lòng bàn tay
        p1, p2, p3 = lm[0], lm[base_idx], lm[base_idx + 1]
        v1, v2 = p2 - p1, p3 - p2
        features.append(vector_2d_angle(v1, v2) / 180.0)
        
        # các ngón dài (Trỏ, Giữa, Áp út, Út)
        if i > 0:
             p1, p2, p3 = lm[base_idx], lm[base_idx+1], lm[base_idx+2]
             v1, v2 = p2 - p1, p3 - p2
             features.append(vector_2d_angle(v1, v2) / 180.0)
        else:
             features.append(0.0) # ngón cái
    
    return features

# Hàm đọc ảnh hỗ trợ nếu đường dẫn có tiếng việt hoặc ký tự đặc biệt
def imread_unicode(path):
    try:
        # Đọc file dưới dạng dữ liệu nhị phân (binary)
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0: return None
        return cv.imdecode(data, cv.IMREAD_COLOR) # Giải mã dữ liệu nhị phân thành ảnh màu
    except Exception: return None

# Hàm duyệt tất cả ảnh trong cấu trúc thư mục dữ liệu
def iter_images(root_dir):
    # Duyệt qua từng thư mục con trong thư mục gốc
    for label_name in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label_name)
        
        # Nếu không phải thư mục thì bỏ qua
        if not os.path.isdir(label_dir):
            continue
        
        # Duyệt qua từng file ảnh trong thư mục nhãn
        for fname in os.listdir(label_dir): 
            if fname.lower().endswith((".jpg", ".jpeg", ".png")): # Chỉ xử lý các file ảnh
                yield label_name, os.path.join(label_dir, fname) 

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True, # xử lý ảnh tĩnh
        max_num_hands=1, # Chỉ lấy 1 bàn tay
        model_complexity=1, # độ phức tạp của mô hình (cân bằng giữa tốc độ và độ chính xác)
        min_detection_confidence=0.3, # Ngưỡng phát hiện bàn tay
    )

    # Kiểm tra thư mục dữ liệu
    if not os.path.isdir(DATA_DIR):
        print(f"Lỗi: Không tìm thấy thư mục dữ liệu {DATA_DIR}")
        return

    
    # Khởi tạo file CSV ('w' - write mode)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Tạo dòng tiêu đề (Header): label, f0, f1, ..., f35
        header = ["label"] + [f"f{i}" for i in range(36)] 
        writer.writerow(header)

        count = 0
        
        # duyệt qua từng ảnh
        for label, path in iter_images(DATA_DIR):
            img = imread_unicode(path)
            if img is None: continue

            rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) # Chuyển ảnh sang hệ màu RGB (do OpenCV đọc ảnh theo hệ BGR)
            res = hands.process(rgb) # Xử lý ảnh để phát hiện bàn tay

            # res.multi_hand_landmarks chứa danh sách các bàn tay tìm được
            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0] # Lấy bàn tay đầu tiên tìm được
                h, w = img.shape[:2] # # Lấy kích thước ảnh gốc (Height, Width)
               
                # Chuyển về tọa độ pixel 
                landmarks_pixel = [[int(lm.x * w), int(lm.y * h)] for lm in hand.landmark]

                # Biến tọa độ pixel thành 36 con số đặc trưng
                features = preprocess_landmarks(landmarks_pixel)
                
                if len(features) == 36: 
                    writer.writerow([label] + features)
                    count += 1
                    if count % 200 == 0: print(f"Đã xử lý {count} ảnh...")

    hands.close()
    print("Hoàn tất!")

if __name__ == "__main__":
    absl_logging.set_verbosity(absl_logging.ERROR)
    main()