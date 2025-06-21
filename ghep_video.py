import cv2

# Đường dẫn đến 4 video đầu vào
video_paths = [
    r"video_camera_view_1.avi",
    r"video_camera_view_2.avi",
    r"video_camera_view_3.avi",
    r"video_camera_view_4.avi"
]

# Mở 4 video
caps = [cv2.VideoCapture(path) for path in video_paths]

# Lấy thông tin kích thước frame và FPS từ video đầu tiên
width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = caps[0].get(cv2.CAP_PROP_FPS)

# Tạo video writer cho đầu ra với kích thước gấp đôi để chứa 4 video nhỏ
out_width = 2 * width
out_height = 2 * height
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output2D.avi', fourcc, fps, (out_width, out_height))

# Xử lý từng frame
while True:
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:  # Nếu không đọc được frame (video đã hết)
            break
        frames.append(frame)
    if len(frames) < 4:  # Nếu không đủ 4 frame, dừng lại
        break

    # Ghép 4 frame thành một frame lớn
    top_row = cv2.hconcat([frames[0], frames[1]])  # Ghép ngang: góc trên trái + trên phải
    bottom_row = cv2.hconcat([frames[2], frames[3]])  # Ghép ngang: góc dưới trái + dưới phải
    out_frame = cv2.vconcat([top_row, bottom_row])  # Ghép dọc: hàng trên + hàng dưới

    # Ghi frame lớn vào video đầu ra
    out.write(out_frame)

# Giải phóng tài nguyên
for cap in caps:
    cap.release()
out.release()
