import os
import shutil
from pathlib import Path

def flatten_image_structure(source_folder, destination_folder, move=False):
    """
    Hàm làm phẳng cấu trúc thư mục ảnh.
    Args:
        source_folder: Đường dẫn folder gốc (TEST1)
        destination_folder: Đường dẫn folder đích (AI)
        move: Nếu True thì sẽ CUT (di chuyển), False thì chỉ COPY.
    """
    
    # Các đuôi file ảnh cần tìm (bạn có thể thêm nếu cần)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    # Kiểm tra folder nguồn có tồn tại không
    if not os.path.exists(source_folder):
        print(f"Lỗi: Thư mục nguồn '{source_folder}' không tồn tại.")
        return

    # Tạo folder đích nếu chưa có
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Lấy danh sách các folder cấp 1 (0, 1, ...)
    # Chỉ lấy folder, bỏ qua file nếu có file nằm ngay trong TEST1
    categories = [d for d in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, d))]

    print(f"Tìm thấy các mục chính: {categories}")

    total_files_processed = 0

    for category in categories:
        src_cat_path = os.path.join(source_folder, category) # VD: TEST1/0
        dst_cat_path = os.path.join(destination_folder, category) # VD: AI/0

        # Tạo folder category trong đích (AI/0, AI/1...)
        os.makedirs(dst_cat_path, exist_ok=True)
        
        print(f"\nĐang xử lý mục: {category}...")
        
        # os.walk sẽ tự động duyệt sâu vào 0.1, Left, Right...
        for root, dirs, files in os.walk(src_cat_path):
            for file in files:
                # Kiểm tra xem có phải file ảnh không
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in valid_extensions:
                    
                    src_file_path = os.path.join(root, file)
                    dst_file_path = os.path.join(dst_cat_path, file)

                    # --- XỬ LÝ TRÙNG TÊN ---
                    # Nếu file đích đã tồn tại (VD: ảnh từ Left có tên giống ảnh từ Right)
                    # Ta sẽ thêm hậu tố vào tên file: IMGxxx.jpg -> IMGxxx_1.jpg
                    counter = 1
                    filename_no_ext = os.path.splitext(file)[0]
                    
                    while os.path.exists(dst_file_path):
                        new_filename = f"{filename_no_ext}_{counter}{file_ext}"
                        dst_file_path = os.path.join(dst_cat_path, new_filename)
                        counter += 1
                    # -----------------------

                    try:
                        if move:
                            shutil.move(src_file_path, dst_file_path)
                        else:
                            shutil.copy2(src_file_path, dst_file_path) # copy2 giữ nguyên metadata (ngày tháng)
                        
                        total_files_processed += 1
                        # In ra cho đỡ buồn nếu chạy nhiều file (tùy chọn)
                        # print(f"Đã {'chuyển' if move else 'copy'}: {file} -> {os.path.basename(dst_file_path)}")
                        
                    except Exception as e:
                        print(f"Lỗi khi xử lý file {file}: {e}")

    print(f"\n--- HOÀN TẤT ---")
    print(f"Tổng cộng đã xử lý {total_files_processed} ảnh.")

# --- CẤU HÌNH CHẠY ---
if __name__ == "__main__":
    # Thay đổi đường dẫn của bạn ở đây
    # Bạn có thể dùng đường dẫn tuyệt đối (C:/Users/...) hoặc tương đối
    FOLDER_NGUON = "TEST1"
    FOLDER_DICH = "AI"
    
    # move=True để cắt file (di chuyển hẳn), move=False để copy (sao chép an toàn)
    print("Bắt đầu quy trình...")
    flatten_image_structure(FOLDER_NGUON, FOLDER_DICH, move=False)