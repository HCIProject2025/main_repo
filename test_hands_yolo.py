import pathlib
pathlib.PosixPath = pathlib.WindowsPath

import cv2
import torch
import numpy as np
import mediapipe as mp
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
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

def process_frame(img0, model, stride, names, device, target_class='bat'):
    h_img, w_img = img0.shape[:2]
    image_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    hand_wrist_positions = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img0, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            x = int(wrist.x * w_img)
            y = int(wrist.y * h_img)
            hand_wrist_positions.append((x, y))
            cv2.putText(img0, 'Hand', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if hand_wrist_positions:
        body_center = (w_img // 2, h_img // 2)
        farthest = max(
            hand_wrist_positions,
            key=lambda pt: np.hypot(pt[0] - body_center[0], pt[1] - body_center[1])
        )
        x_line = farthest[0]
        cv2.line(img0, (x_line, 0), (x_line, h_img), (0, 255, 255), 2)

    # Object Detection
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

    for det in pred:  # det는 [N×6] 형태의 tensor (N = 검출된 박스 개수)
        if det is None or det.shape[0] == 0:
            continue
        if class_index is not None:
            det = det[det[:, -1] == class_index]  # 클래스 필터링
        if det.shape[0] == 0:
            continue

        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

        # confidence 기준으로 가장 높은 것 한 개만 남기기
        if det.shape[0] > 1:
            best_idx = det[:, 4].argmax().unsqueeze(0)
            det = det[best_idx]

        # 한 개 혹은 여러 개의 row가 남은 det: det.shape = [M,6]
        for row in det:
            x1, y1, x2, y2 = map(int, row[:4].tolist())
            conf = float(row[4].item())
            cls = int(row[5].item())

            if conf < 0.3:
                continue

            # ─── 박스 그리기 주석 처리 ───────────────────────────────────
            # label = f'{names[cls]} {conf:.2f}'
            # cv2.rectangle(img0, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.putText(img0, label, (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            # ──────────────────────────────────────────────────────────

            # 배트 영역에서 선(작대기) 검출
            bat_crop = img0[y1:y2, x1:x2].copy()
            gray = cv2.cvtColor(bat_crop, cv2.COLOR_BGR2GRAY)
            blurred = cv2.bilateralFilter(gray, 9, 75, 75)
            edges = cv2.Canny(blurred, 30, 100)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                    threshold=20, minLineLength=20, maxLineGap=10)
            if lines is not None:
                # 가장 긴 선분만 선택
                longest = max(lines, key=lambda l: np.linalg.norm(
                    np.array(l[0][:2]) - np.array(l[0][2:])))
                x1_, y1_, x2_, y2_ = longest[0]
                pt1 = (x1_ + x1, y1_ + y1)
                pt2 = (x2_ + x1, y2_ + y1)

                # 연장 로직: 가장 긴 선분을 화면 끝까지 연장
                direction = np.array(pt2) - np.array(pt1)
                unit_vec = direction / np.linalg.norm(direction)
                length = max(h_img, w_img) * 2  # 충분히 큰 값을 곱해 연장

                extended_pt1 = (
                    int(pt1[0] - unit_vec[0] * length),
                    int(pt1[1] - unit_vec[1] * length)
                )
                extended_pt2 = (
                    int(pt2[0] + unit_vec[0] * length),
                    int(pt2[1] + unit_vec[1] * length)
                )

                cv2.line(img0, extended_pt1, extended_pt2, (0, 255, 0), 2)

                # 손목과 선 끝점을 연결해서 표시하고 싶다면 아래 주석 해제
                # if hand_wrist_positions:
                #     hand = np.mean(hand_wrist_positions, axis=0)
                #     d1 = np.linalg.norm(np.array(hand) - np.array(pt1))
                #     d2 = np.linalg.norm(np.array(hand) - np.array(pt2))
                #     head_pt = tuple(map(int, pt1 if d1 > d2 else pt2))
                #     cv2.circle(img0, head_pt, 6, (0, 0, 255), -1)

    return img0

def detect_video_vertical(video_path, target_class='bat', weights='runs/train/final_learn/weights/best.pt',
                          rotate=False):
    """
    video_path: 입력 비디오 경로
    target_class: 감지할 클래스 이름 (기본 'bat')
    weights: YOLOv5 가중치 파일 경로
    rotate: True일 경우 프레임을 시계 방향 90도 회전
    """
    model, stride, names, device = load_model(weights)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ 비디오 파일을 열 수 없습니다: {video_path}")
        return

    # 출력 해상도 설정 (원하는 크기로 높이/너비 설정)
    output_width, output_height = 1280, 900

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 90도 회전 옵션
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # 해상도 높이기: 원본보다 큰 크기로 확대
        frame = cv2.resize(frame, (output_width, output_height))

        output = process_frame(frame, model, stride, names, device, target_class)
        cv2.imshow('Vertical Video Detection', output)

        # 재생 속도 조절: 100ms 대기 → 약 10FPS
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # rotate=True로 주면 90도 회전 후 처리
    detect_video_vertical('video/swing_test.mp4', rotate=True)
