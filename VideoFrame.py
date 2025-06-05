import os

import cv2

video_path = "video/swing_test.mp4"  # 🎞️ 동영상 경로
output_dir = "swing_test_frames"  # 🖼️ 저장할 폴더
interval_sec = 0.2  # 📸 몇 초마다 프레임 추출할지

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * interval_sec)

count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if count % frame_interval == 0:
        filename = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        saved += 1
    count += 1

cap.release()
print("✅ 프레임 추출 완료!")
