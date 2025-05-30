import math

import cv2
import torch

# YOLO 모델 경로
MODEL_PATH = "../runs/train/bat_detector3/weights/best.pt"
# 모델 로드
yolo = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True)


def angle_from_horizontal(v):
    ref = (1, 0)  # 기준 수평 벡터
    dot = v[0] * ref[0] + v[1] * ref[1]
    mag_v = math.hypot(v[0], v[1])
    if mag_v == 0:
        return 0
    angle = math.acos(dot / mag_v)
    return math.degrees(angle)


# 웹캠 영상 열기 (0번 카메라)
cap = cv2.VideoCapture(5)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame)
    boxes = results.xyxy[0].cpu().numpy()

    swing = False
    label = ""

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        if int(cls) == 0:  # 배트 클래스라고 가정
            base = (int(x1), int((y1 + y2) / 2))
            head = (int(x2), int((y1 + y2) / 2))

            bat_vec = (head[0] - base[0], head[1] - base[1])
            angle = angle_from_horizontal(bat_vec)

            swing = angle > 30  # 기준 각도는 조정 가능
            label = f"SWING! ({angle:.1f}°)" if swing else f"NO SWING ({angle:.1f}°)"

            # 시각화
            cv2.line(frame, base, head, (0, 255, 0), 4)
            cv2.circle(frame, head, 8, (0, 0, 255), -1)

    # 텍스트 표시
    if label:
        color = (0, 255, 0) if swing else (0, 0, 255)
        cv2.putText(frame, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

    cv2.imshow("Swing Angle Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
