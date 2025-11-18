# # build_dataset.py
# import os

# # Reduce TensorFlow/absl/glog verbosity before importing libraries that load TF
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
# os.environ["GLOG_minloglevel"] = "2"      # Silence warning logs from native libs
# os.environ["ABSL_MIN_LOG_LEVEL"] = "2"    # absl logging level
# # If you prefer to disable oneDNN to avoid numerical variance, uncomment:
# # os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# import csv
# import itertools
# import unicodedata
# import cv2 as cv
# import numpy as np
# from absl import logging as absl_logging
# import mediapipe as mp

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# # Point to TEST1 next to this script by default; change to an absolute path if needed
# DATA_DIR = os.path.join(SCRIPT_DIR, "TEST1")  # root with subfolders per class
# OUT_CSV = "keypoints_dataset.csv"

# def preprocess_landmarks(landmarks_xy):
#     # landmarks_xy: list of 21 [x, y] pixel coords
#     lm = np.array(landmarks_xy, dtype=np.float32)
#     base = lm[0].copy()  # wrist as origin
#     lm -= base
#     lm = lm.flatten()
#     max_val = np.max(np.abs(lm)) if np.max(np.abs(lm)) > 0 else 1.0
#     lm = lm / max_val
#     return lm.tolist()

# def normalize_side_name(name):
#     # Map Vietnamese/English folder names to canonical 'Left'/'Right'
#     # Handles accents and case differences.
#     normalized = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
#     normalized = normalized.strip().lower()
#     if normalized in {"trai", "left", "l"}:
#         return "Left"
#     if normalized in {"phai", "phai ", "right", "r"}:
#         return "Right"
#     # Fallback: title-case original without accents
#     return normalized.title()

# def iter_images(root_dir):
#     # Expect structure: root/label/(Phải|Trái)/image files
#     for label_name in sorted(os.listdir(root_dir)):
#         label_dir = os.path.join(root_dir, label_name)
#         if not os.path.isdir(label_dir):
#             continue
#         # Iterate side subfolders if present; otherwise, treat files directly under label
#         has_side_dirs = False
#         for side_name in sorted(os.listdir(label_dir)):
#             side_dir = os.path.join(label_dir, side_name)
#             if not os.path.isdir(side_dir):
#                 continue
#             has_side_dirs = True
#             side_canonical = normalize_side_name(side_name)
#             composed_label = f"{label_name}_{side_canonical}"
#             for fname in os.listdir(side_dir):
#                 if fname.lower().endswith((".jpg", ".jpeg", ".png")):
#                     yield composed_label, os.path.join(side_dir, fname)
#         if not has_side_dirs:
#             # No side directories; yield images directly with the original label
#             for fname in os.listdir(label_dir):
#                 if fname.lower().endswith((".jpg", ".jpeg", ".png")):
#                     yield label_name, os.path.join(label_dir, fname)

# def main():
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(
#         static_image_mode=True,
#         max_num_hands=2,
#         model_complexity=1,
#         min_detection_confidence=0.3,
#     )

#     # Quick sanity check and discovery summary
#     if not os.path.isdir(DATA_DIR):
#         raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

#     total_images = 0
#     discovered_labels = set()
#     for label, path in iter_images(DATA_DIR):
#         discovered_labels.add(label)
#         total_images += 1
#     print(f"Found {total_images} images across {len(discovered_labels)} labels in {DATA_DIR}")

#     with open(OUT_CSV, "w", newline="") as f:
#         writer = csv.writer(f)
#         # header: label + 42 features (21*2)
#         header = ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
#         writer.writerow(header)

#         detected_count = 0
#         skipped_count = 0
#         failed_samples = []
#         for label, path in iter_images(DATA_DIR):
#             img = cv.imread(path)
#             if img is None:
#                 skipped_count += 1
#                 failed_samples.append((label, path, "imread_failed"))
#                 continue
#             rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#             res = hands.process(rgb)
#             if not res.multi_hand_landmarks:
#                 skipped_count += 1
#                 if len(failed_samples) < 50:
#                     failed_samples.append((label, path, "no_hand"))
#                 continue
#             hand = res.multi_hand_landmarks[0]
#             h, w = img.shape[:2]
#             landmarks = []
#             for lm in hand.landmark:
#                 landmarks.append([int(lm.x * w), int(lm.y * h)])

#             features = preprocess_landmarks(landmarks)
#             writer.writerow([label] + features)
#             detected_count += 1

#     # Write a quick diagnostics file for skipped images
#     diag_path = os.path.join(SCRIPT_DIR, "build_dataset_diagnostics.txt")
#     with open(diag_path, "w", encoding="utf-8") as df:
#         df.write(f"DATA_DIR: {DATA_DIR}\n")
#         df.write(f"Detected: {detected_count}\n")
#         df.write(f"Skipped: {skipped_count}\n")
#         for lab, pth, reason in failed_samples:
#             df.write(f"{reason}\t{lab}\t{pth}\n")

#     hands.close()
#     print(f"Saved dataset to {OUT_CSV}")

# if __name__ == "__main__":
#     # Ensure absl only reports errors to avoid STDERR warnings
#     absl_logging.set_verbosity(absl_logging.ERROR)
#     main()




# import os  # Thư viện thao tác với hệ thống tệp và biến môi trường

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Ẩn log INFO của TensorFlow
# os.environ["GLOG_minloglevel"] = "2"      # Ẩn log INFO của glog
# os.environ["ABSL_MIN_LOG_LEVEL"] = "2"    # Ẩn log INFO của absl

# import csv  # Đọc/ghi file CSV
# import unicodedata  # Chuẩn hóa chuỗi Unicode (loại bỏ dấu tiếng Việt)
# import numpy as np  # Thư viện xử lý mảng số
# import cv2 as cv  # OpenCV để xử lý ảnh
# import mediapipe as mp  # Thư viện nhận diện landmark tay
# from absl import logging as absl_logging  # Giao diện logging của absl

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Thư mục chứa file script
# DATA_DIR = os.path.join(SCRIPT_DIR, "TEST1")  # Thư mục chứa dữ liệu ảnh
# OUT_CSV = "keypoints_dataset.csv"  # Tên file CSV đầu ra


# def imread_unicode(path):  # Đọc ảnh từ đường dẫn Unicode (hữu ích trên Windows)
#     try:
#         data = np.fromfile(path, dtype=np.uint8)  # Đọc file nhị phân thành mảng byte
#         if data.size == 0:
#             return None  # Trả về None nếu file rỗng
#         return cv.imdecode(data, cv.IMREAD_COLOR)  # Giải mã ảnh từ mảng byte
#     except Exception:
#         return None  # Trả về None nếu có lỗi


# def videocapture_unicode(path):  # Mở video từ đường dẫn Unicode (hữu ích trên Windows)
#     # cv2.VideoCapture có vấn đề với Unicode paths trên Windows
#     # Thử mở với path gốc trước
#     cap = cv.VideoCapture(path)
#     if cap.isOpened():
#         return cap
    
#     # Nếu thất bại, thử với encoding khác hoặc short path (Windows)
#     try:
#         import sys
#         if sys.platform == "win32":
#             # Trên Windows, thử dùng short path name
#             try:
#                 import ctypes
#                 from ctypes import wintypes
#                 kernel32 = ctypes.windll.kernel32
#                 GetShortPathNameW = kernel32.GetShortPathNameW
#                 GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
#                 GetShortPathNameW.restype = wintypes.DWORD
                
#                 # Chuyển đổi sang short path
#                 short_path = ctypes.create_unicode_buffer(260)
#                 result = GetShortPathNameW(path, short_path, 260)
#                 if result > 0:
#                     cap = cv.VideoCapture(short_path.value)
#                     if cap.isOpened():
#                         return cap
#             except Exception:
#                 pass
            
#             # Thử với path được encode
#             try:
#                 encoded_path = path.encode("utf-8").decode("utf-8")
#                 cap = cv.VideoCapture(encoded_path)
#                 if cap.isOpened():
#                     return cap
#             except Exception:
#                 pass
#     except Exception:
#         pass
    
#     # Trả về cap đã tạo (có thể không mở được)
#     return cap


# def normalize_side_name(name):  # Chuẩn hóa tên thư mục tay trái/phải
#     normalized = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")  # Bỏ dấu tiếng Việt
#     normalized = normalized.strip().lower()  # Bỏ khoảng trắng và chuyển về chữ thường
#     if normalized in {"trai", "left", "l"}:
#         return "Left"  # Gán nhãn tay trái
#     if normalized in {"phai", "right", "r"}:
#         return "Right"  # Gán nhãn tay phải
#     return normalized.title()  # Trả về tên viết hoa chữ cái đầu nếu không khớp



# def iter_images(root_dir):  # Duyệt toàn bộ ảnh trong thư mục gốc
#     # Cấu trúc: root/gesture_name/part_name/Left|Right/images
#     # Nhãn cuối cùng: gesture_name_Left hoặc gesture_name_Right (bỏ qua part_name)
#     for gesture_name in sorted(os.listdir(root_dir)):  # Duyệt từng thư mục gesture
#         gesture_dir = os.path.join(root_dir, gesture_name)
#         if not os.path.isdir(gesture_dir):
#             continue  # Bỏ qua nếu không phải thư mục

#         # Duyệt các thư mục part (front, back, tilt, 0, 1, 2...)
#         for part_name in sorted(os.listdir(gesture_dir)):
#             part_dir = os.path.join(gesture_dir, part_name)
#             if not os.path.isdir(part_dir):
#                 continue

#             # Kiểm tra xem có thư mục Left/Right không
#             has_side_dirs = False
#             for side_name in sorted(os.listdir(part_dir)):
#                 side_dir = os.path.join(part_dir, side_name)
#                 if not os.path.isdir(side_dir):
#                     continue
#                 has_side_dirs = True
#                 side_canonical = normalize_side_name(side_name)  # Chuẩn hóa tên tay
#                 composed_label = f"{gesture_name}_{side_canonical}"  # Nhãn: gesture_Left hoặc gesture_Right

#                 # Duyệt ảnh trong thư mục tay
#                 for fname in os.listdir(side_dir):
#                     if fname.lower().endswith((".jpg", ".jpeg", ".png")):
#                         yield composed_label, os.path.join(side_dir, fname)

#             # Nếu không có thư mục Left/Right, kiểm tra xem có ảnh trực tiếp trong part_dir không
#             if not has_side_dirs:
#                 for fname in os.listdir(part_dir):
#                     if fname.lower().endswith((".jpg", ".jpeg", ".png")):
#                         # Nếu không có Left/Right, dùng nhãn gesture_name
#                         yield gesture_name, os.path.join(part_dir, fname)


# def iter_videos(root_dir):  # Duyệt toàn bộ video trong thư mục gốc
#     # Cấu trúc: root/gesture_name/video/Left|Right/video files
#     # Nhãn cuối cùng: gesture_name_Left hoặc gesture_name_Right
#     for gesture_name in sorted(os.listdir(root_dir)):  # Duyệt từng thư mục gesture
#         gesture_dir = os.path.join(root_dir, gesture_name)
#         if not os.path.isdir(gesture_dir):
#             continue  # Bỏ qua nếu không phải thư mục

#         # Kiểm tra xem có thư mục "video" trực tiếp trong gesture_dir không
#         video_dir = os.path.join(gesture_dir, "video")
#         if not os.path.isdir(video_dir):
#             continue  # Không có thư mục video, bỏ qua

#         # Duyệt các thư mục Left/Right trong thư mục video
#         for side_name in sorted(os.listdir(video_dir)):
#             side_dir = os.path.join(video_dir, side_name)
#             if not os.path.isdir(side_dir):
#                 continue
#             side_canonical = normalize_side_name(side_name)  # Chuẩn hóa tên tay
#             composed_label = f"{gesture_name}_{side_canonical}"  # Nhãn: gesture_Left hoặc gesture_Right

#             # Duyệt video trong thư mục tay
#             for fname in os.listdir(side_dir):
#                 if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm")):
#                     yield composed_label, os.path.join(side_dir, fname)


# def extract_frames_from_video(video_path, frame_interval=1):  # Trích xuất frame từ video
#     # frame_interval: lấy frame mỗi N giây (mặc định là 1 giây)
#     cap = videocapture_unicode(video_path)  # Sử dụng hàm hỗ trợ Unicode path
#     if not cap.isOpened():
#         return []  # Không thể mở video, trả về danh sách rỗng
    
#     fps = cap.get(cv.CAP_PROP_FPS)  # Lấy FPS của video
#     if fps <= 0:
#         fps = 30  # Giá trị mặc định nếu không lấy được FPS
    
#     frames = []
#     frame_count = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break  # Hết video
        
#         # Lấy frame mỗi frame_interval giây
#         if frame_count % int(fps * frame_interval) == 0:
#             frames.append(frame.copy())
        
#         frame_count += 1
    
#     cap.release()
#     return frames


# def preprocess_landmarks(landmarks_xy):  # Chuẩn hóa và scale tọa độ landmark
#     lm = np.array(landmarks_xy, dtype=np.float32)  # Chuyển sang mảng float32
#     base = lm[0].copy()  # Lấy cổ tay làm gốc tọa độ
#     lm -= base  # Dời toàn bộ điểm về gốc
#     max_val = np.max(np.abs(lm)) if np.max(np.abs(lm)) > 0 else 1.0  # Tìm giá trị lớn nhất để scale
#     lm = lm / max_val  # Scale về [-1, 1]
#     xs = lm[:, 0].tolist()  # Tách danh sách x
#     ys = lm[:, 1].tolist()  # Tách danh sách y
#     return xs + ys  # Trả về dạng [x0..x20, y0..y20]

# def main():
#     mp_hands = mp.solutions.hands  # Khởi tạo mô-đun nhận diện tay
#     hands = mp_hands.Hands(
#         static_image_mode=True,  # Chế độ ảnh tĩnh
#         max_num_hands=1,  # Chỉ nhận 1 tay mỗi ảnh
#         model_complexity=1,  # Độ phức tạp của mô hình
#         min_detection_confidence=0.3,  # Ngưỡng phát hiện tay
#     )

#     if not os.path.isdir(DATA_DIR):
#         raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")  # Báo lỗi nếu không tìm thấy thư mục

#     total_images = 0
#     total_videos = 0
#     discovered_labels = set()
#     for label, path in iter_images(DATA_DIR):  # Đếm số ảnh và nhãn
#         discovered_labels.add(label)
#         total_images += 1
#     for label, path in iter_videos(DATA_DIR):  # Đếm số video và nhãn
#         discovered_labels.add(label)
#         total_videos += 1
#     print(f"Found {total_images} images and {total_videos} videos across {len(discovered_labels)} labels in {DATA_DIR}")

#     with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:  # Mở file CSV để ghi
#         writer = csv.writer(f)
#         header = ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]  # Tạo header
#         writer.writerow(header)

#         detected_count = 0
#         skipped_count = 0
#         failed_samples = []

#         # Xử lý ảnh
#         for label, path in iter_images(DATA_DIR):  # Duyệt từng ảnh
#             img = imread_unicode(path)  # Đọc ảnh
#             if img is None:
#                 skipped_count += 1
#                 failed_samples.append((label, path, "imread_failed"))
#                 continue

#             rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Chuyển ảnh sang RGB
#             res = hands.process(rgb)  # Nhận diện tay

#             if not res.multi_hand_landmarks:  # Không phát hiện tay
#                 skipped_count += 1
#                 if len(failed_samples) < 50:
#                     failed_samples.append((label, path, "no_hand"))
#                 continue

#             hand = res.multi_hand_landmarks[0]  # Lấy tay đầu tiên
#             h, w = img.shape[:2]
#             landmarks = [[int(lm.x * w), int(lm.y * h)] for lm in hand.landmark]  # Chuyển tọa độ về pixel

#             features = preprocess_landmarks(landmarks)  # Chuẩn hóa landmark
#             if len(features) != 42:  # Kiểm tra số lượng đặc trưng
#                 skipped_count += 1
#                 failed_samples.append((label, path, f"bad_feature_len:{len(features)}"))
#                 continue

#             writer.writerow([label] + features)  # Ghi dòng dữ liệu vào CSV
#             detected_count += 1

#         # Xử lý video
#         for label, video_path in iter_videos(DATA_DIR):  # Duyệt từng video
#             frames = extract_frames_from_video(video_path, frame_interval=1)  # Trích xuất frame (mỗi 1 giây)
#             if not frames:
#                 skipped_count += 1
#                 failed_samples.append((label, video_path, "video_extract_failed"))
#                 continue

#             video_detected = False
#             for frame_idx, frame in enumerate(frames):
#                 rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Chuyển frame sang RGB
#                 res = hands.process(rgb)  # Nhận diện tay

#                 if not res.multi_hand_landmarks:  # Không phát hiện tay trong frame này
#                     continue  # Bỏ qua frame này, thử frame tiếp theo

#                 hand = res.multi_hand_landmarks[0]  # Lấy tay đầu tiên
#                 h, w = frame.shape[:2]
#                 landmarks = [[int(lm.x * w), int(lm.y * h)] for lm in hand.landmark]  # Chuyển tọa độ về pixel

#                 features = preprocess_landmarks(landmarks)  # Chuẩn hóa landmark
#                 if len(features) != 42:  # Kiểm tra số lượng đặc trưng
#                     continue  # Bỏ qua frame này

#                 writer.writerow([label] + features)  # Ghi dòng dữ liệu vào CSV
#                 detected_count += 1
#                 video_detected = True

#             if not video_detected:  # Không có frame nào phát hiện được tay
#                 skipped_count += 1
#                 if len(failed_samples) < 50:
#                     failed_samples.append((label, video_path, "no_hand_in_video"))
            
#     diag_path = os.path.join(SCRIPT_DIR, "build_dataset_diagnostics.txt")  # Đường dẫn file log
#     with open(diag_path, "w", encoding="utf-8") as df:
#         df.write(f"DATA_DIR: {DATA_DIR}\n")
#         df.write(f"Detected: {detected_count}\n")
#         df.write(f"Skipped: {skipped_count}\n")
#         for lab, pth, reason in failed_samples:
#             df.write(f"{reason}\t{lab}\t{pth}\n")  # Ghi lý do lỗi và đường dẫn ảnh

#     hands.close()  # Giải phóng tài nguyên MediaPipe
#     print(f"Saved dataset to {OUT_CSV}")  # Thông báo hoàn tất
#     print(f"Diagnostics written to {diag_path}")


# if __name__ == "__main__":
#     absl_logging.set_verbosity(absl_logging.ERROR)  # Tắt log absl
#     main()  # Gọi hàm chính



# build_dataset.py (Phiên bản Cải tiến)
# build_dataset.py (Phiên bản Đã Sửa Lỗi Đọc Ảnh Mới)
import os
import csv
import unicodedata
import numpy as np
import cv2 as cv
import mediapipe as mp
from absl import logging as absl_logging
import ctypes
from ctypes import wintypes

# Cấu hình log
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"
os.environ["ABSL_MIN_LOG_LEVEL"] = "2"

# --- Cấu hình ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "TEST1")
OUT_CSV = "keypoints_dataset.csv"
# --- End Cấu hình ---


# --- HÀM HỖ TRỢ ĐỌC FILE/VIDEO UNICODE (ĐÃ SỬA LỖI) ---

def imread_unicode(path):
    """
    Đọc ảnh hỗ trợ đường dẫn Unicode. 
    Nếu np.fromfile thất bại, thử dùng cv.imread tiêu chuẩn.
    """
    try:
        # Cách 1: Đọc bằng np.fromfile (tốt nhất cho Unicode)
        data = np.fromfile(path, dtype=np.uint8)
        if data is not None and data.size > 0:
            return cv.imdecode(data, cv.IMREAD_COLOR)
        return None
    except Exception:
        # Cách 2: Fallback (thường không hoạt động với Unicode, nhưng để đảm bảo)
        try:
             img = cv.imread(path)
             if img is not None and img.size > 0:
                 return img
             return None
        except Exception:
            return None

def videocapture_unicode(path):
    # Giữ nguyên hàm xử lý video, vì logic này thường ổn định
    cap = cv.VideoCapture(path)
    if cap.isOpened():
        return cap
    
    try:
        if os.name == "nt": # Dùng GetShortPathNameW cho Windows
            kernel32 = ctypes.windll.kernel32
            GetShortPathNameW = kernel32.GetShortPathNameW
            GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
            GetShortPathNameW.restype = wintypes.DWORD
            short_path = ctypes.create_unicode_buffer(260)
            result = GetShortPathNameW(path, short_path, 260)
            if result > 0:
                cap = cv.VideoCapture(short_path.value)
                if cap.isOpened():
                    return cap
    except Exception:
        pass
    return cap

# --- HÀM HỖ TRỢ XỬ LÝ DỮ LIỆU (Duy trì) ---

def normalize_side_name(name):
    normalized = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.strip().lower()
    if normalized in {"trai", "left", "l"}:
        return "Left"
    if normalized in {"phai", "right", "r"}:
        return "Right"
    return normalized.title()

def is_image_file(fname):
    """Kiểm tra xem tên file có phải là định dạng ảnh hợp lệ không."""
    return fname.lower().endswith((".jpg", ".jpeg", ".png"))

def is_video_file(fname):
    """Kiểm tra xem tên file có phải là định dạng video hợp lệ không."""
    return fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"))


def iter_images(root_dir):
    """
    Duyệt structure:
    TEST1/
      gesture/
         part/
             Left/*.jpg
             Right/*.jpg
         part2/
             ảnh trực tiếp (không có Left/Right)
    """
    for gesture_name in sorted(os.listdir(root_dir)):
        gesture_dir = os.path.join(root_dir, gesture_name)
        if not os.path.isdir(gesture_dir):
            continue

        for part_name in sorted(os.listdir(gesture_dir)):
            part_dir = os.path.join(gesture_dir, part_name)
            if not os.path.isdir(part_dir):
                continue

            # Kiểm tra nếu trong part có Left / Right
            left_dir = os.path.join(part_dir, "Left")
            right_dir = os.path.join(part_dir, "Right")

            if os.path.isdir(left_dir):
                for fname in os.listdir(left_dir):
                    if is_image_file(fname):
                        yield f"{gesture_name}_Left", os.path.join(left_dir, fname).replace("\\", "/")

            if os.path.isdir(right_dir):
                for fname in os.listdir(right_dir):
                    if is_image_file(fname):
                        yield f"{gesture_name}_Right", os.path.join(right_dir, fname).replace("\\", "/")

            # Nếu part không có Left/Right → ảnh nằm trực tiếp (structure 3 cấp)
            if not os.path.isdir(left_dir) and not os.path.isdir(right_dir):
                for fname in os.listdir(part_dir):
                    if is_image_file(fname):
                        yield gesture_name, os.path.join(part_dir, fname).replace("\\", "/")

def iter_videos(root_dir):
    # Giữ nguyên hàm duyệt video
    for gesture_name in sorted(os.listdir(root_dir)):
        gesture_dir = os.path.join(root_dir, gesture_name)
        if not os.path.isdir(gesture_dir): continue

        video_dir = os.path.join(gesture_dir, "video")
        if not os.path.isdir(video_dir): continue

        for side_name in sorted(os.listdir(video_dir)):
            side_dir = os.path.join(video_dir, side_name)
            if not os.path.isdir(side_dir): continue
            side_canonical = normalize_side_name(side_name)
            composed_label = f"{gesture_name}_{side_canonical}"

            for fname in os.listdir(side_dir):
                if is_video_file(fname):
                    yield composed_label, os.path.join(side_dir, fname)

def extract_frames_from_video(video_path, frame_interval=1):
    cap = videocapture_unicode(video_path)
    if not cap.isOpened(): return []
    
    fps = cap.get(cv.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_count % int(fps * frame_interval) == 0:
            frames.append(frame.copy())
        
        frame_count += 1
    
    cap.release()
    return frames

# --- HÀM XỬ LÝ ĐẶC TRƯNG CẢI TIẾN (Giữ nguyên) ---

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
        
    return np.array(features, dtype=np.float32).tolist()


# --- HÀM MAIN (Giữ nguyên logic) ---

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.3,
    )
    
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

    total_images = 0
    total_videos = 0
    discovered_labels = set()
    
    # Đếm số lượng để hiển thị thông báo đầu ra
    for label, path in iter_images(DATA_DIR):
        discovered_labels.add(label)
        total_images += 1
    for label, path in iter_videos(DATA_DIR):
        discovered_labels.add(label)
        total_videos += 1
    print(f"Found {total_images} images and {total_videos} videos across {len(discovered_labels)} labels in {DATA_DIR}")

    EXPECTED_FEATURE_LENGTH = 36 

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        header = ["label"] + [f"feature_{i}" for i in range(EXPECTED_FEATURE_LENGTH)]
        writer.writerow(header)

        detected_count = 0
        skipped_count = 0
        failed_samples = []

        # Xử lý ảnh
        for label, path in iter_images(DATA_DIR):
            img = imread_unicode(path)
            if img is None:
                skipped_count += 1
                failed_samples.append((label, path, "imread_failed"))
                continue

            rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if not res.multi_hand_landmarks:
                skipped_count += 1
                if len(failed_samples) < 50:
                    failed_samples.append((label, path, "no_hand"))
                continue

            hand = res.multi_hand_landmarks[0]
            h, w = img.shape[:2]
            landmarks = [[int(lm.x * w), int(lm.y * h)] for lm in hand.landmark] 

            features = preprocess_landmarks(landmarks)
            if len(features) != EXPECTED_FEATURE_LENGTH:
                skipped_count += 1
                failed_samples.append((label, path, f"bad_feature_len:{len(features)}"))
                continue

            writer.writerow([label] + features)
            detected_count += 1

        # Xử lý video (giữ nguyên)
        for label, video_path in iter_videos(DATA_DIR):
            frames = extract_frames_from_video(video_path, frame_interval=1)
            if not frames:
                skipped_count += 1
                failed_samples.append((label, video_path, "video_extract_failed"))
                continue

            video_detected = False
            for frame_idx, frame in enumerate(frames):
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                res = hands.process(rgb)

                if not res.multi_hand_landmarks: continue

                hand = res.multi_hand_landmarks[0]
                h, w = frame.shape[:2]
                landmarks = [[int(lm.x * w), int(lm.y * h)] for lm in hand.landmark]

                features = preprocess_landmarks(landmarks)
                if len(features) != EXPECTED_FEATURE_LENGTH: continue

                writer.writerow([label] + features)
                detected_count += 1
                video_detected = True

            if not video_detected:
                skipped_count += 1
                if len(failed_samples) < 50:
                    failed_samples.append((label, video_path, "no_hand_in_video"))
            
    diag_path = os.path.join(SCRIPT_DIR, "build_dataset_diagnostics.txt")
    with open(diag_path, "w", encoding="utf-8") as df:
        df.write(f"DATA_DIR: {DATA_DIR}\n")
        df.write(f"Detected: {detected_count}\n")
        df.write(f"Skipped: {skipped_count}\n")
        for lab, pth, reason in failed_samples:
            df.write(f"{reason}\t{lab}\t{pth}\n")

    hands.close()
    print(f"Saved dataset to {OUT_CSV}. **Features length: {EXPECTED_FEATURE_LENGTH}**")
    print(f"Diagnostics written to {diag_path}")


if __name__ == "__main__":
    absl_logging.set_verbosity(absl_logging.ERROR)
    main()