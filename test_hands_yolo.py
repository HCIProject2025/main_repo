import cv2
import torch
import numpy as np
import mediapipe as mp
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def load_model(weights='yolov5s.pt', device=''):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride = model.stride
    names = model.names
    return model, stride, names, device

def detect_image(img_path, target_class='baseball bat', weights='yolov5s.pt'):
    model, stride, names, device = load_model(weights)
    img0 = cv2.imread(img_path)

    if img0 is None:
        print(f"⚠️ 이미지 파일을 불러올 수 없습니다: {img_path}")
        return

    image_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    hand_wrist_positions = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img0,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            x = int(wrist.x * img0.shape[1])
            y = int(wrist.y * img0.shape[0])
            hand_wrist_positions.append((x, y))
            cv2.putText(img0, 'Hand', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if len(hand_wrist_positions) >= 1:
        leftmost_hand = min(hand_wrist_positions, key=lambda pt: pt[0])
        x = leftmost_hand[0]
        cv2.line(img0, (x, 0), (x, img0.shape[0]), (0, 255, 255), 2)

    img = letterbox(img0, stride=stride)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)
    pred = non_max_suppression(pred, 0.25, 0.45)

    class_index = next((i for i, name in names.items() if name == target_class), None)

    found_bat = False
    for det in pred:
        if det is not None and len(det):
            if class_index is not None:
                det = det[det[:, -1] == class_index]
            if len(det) == 0:
                continue
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img0, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # 배트 crop 후 선 추출
                bat_crop = img0[y1:y2, x1:x2].copy()
                gray = cv2.cvtColor(bat_crop, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=5)

                if lines is not None:
                    longest = max(lines, key=lambda l: np.linalg.norm(np.array(l[0][:2]) - np.array(l[0][2:])))
                    x1_, y1_, x2_, y2_ = longest[0]
                    pt1 = (x1_ + x1, y1_ + y1)
                    pt2 = (x2_ + x1, y2_ + y1)
                    cv2.line(img0, pt1, pt2, (0, 255, 0), 2)

                    # 손 기준 거리 계산 → 헤드 표시
                    if hand_wrist_positions:
                        hand = np.mean(hand_wrist_positions, axis=0)
                        d1 = np.linalg.norm(np.array(hand) - np.array(pt1))
                        d2 = np.linalg.norm(np.array(hand) - np.array(pt2))
                        head_pt = tuple(map(int, pt1 if d1 > d2 else pt2))
                        cv2.circle(img0, head_pt, 6, (0, 0, 255), -1)
            found_bat = True

    if not found_bat and not hand_wrist_positions:
        print("⚠️ 배트와 손 모두 감지되지 않음!")
    cv2.imshow('Detection', img0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 실행 예시
detect_image('video/swing1.png')
