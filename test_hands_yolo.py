import pathlib
import time

# PosixPath ↔ WindowsPath 패치
pathlib.PosixPath = pathlib.WindowsPath

import cv2
import mediapipe as mp
import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
)


def load_model(weights="runs/train/final_learn3/weights/best.pt", device=""):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride = model.stride
    names = model.names
    return model, stride, names, device


# 전역 변수: 마지막 프레임에서 계산된 배트 선분(확장된) 좌표
bat_line_points = None


def process_frame(img0, model, stride, names, device, target_class="bat"):
    global bat_line_points
    bat_line_points = None

    h_img, w_img = img0.shape[:2]
    image_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    hand_wrist_positions = []
    current_hand_pos = None

    # 1) 손 모양(랜드마크) 검출
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img0,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
            )
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            x = int(wrist.x * w_img)
            y = int(wrist.y * h_img)
            hand_wrist_positions.append((x, y))
            current_hand_pos = (x, y)
            cv2.putText(img0, "Hand", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 2) 배트 검출 (YOLOv5) + 선(Hough) + 연장
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

    for det in pred:
        if det is None or det.shape[0] == 0:
            continue
        if class_index is not None:
            det = det[det[:, -1] == class_index]
        if det.shape[0] == 0:
            continue

        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

        # 가장 높은 confidence 하나만 남기기
        if det.shape[0] > 1:
            best_idx = det[:, 4].argmax().unsqueeze(0)
            det = det[best_idx]

        for row in det:
            x1, y1, x2, y2 = map(int, row[:4].tolist())
            conf = float(row[4].item())

            if conf < 0.1:
                continue

            # 배트 영역에서 Hough 선 검출 & 연장
            bat_crop = img0[y1:y2, x1:x2].copy()
            gray = cv2.cvtColor(bat_crop, cv2.COLOR_BGR2GRAY)
            blurred = cv2.bilateralFilter(gray, 9, 75, 75)
            edges = cv2.Canny(blurred, 30, 100)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=10)

            if lines is not None:
                longest = max(lines, key=lambda l: np.linalg.norm(np.array(l[0][:2]) - np.array(l[0][2:])))
                x1_, y1_, x2_, y2_ = longest[0]
                pt1 = (x1_ + x1, y1_ + y1)
                pt2 = (x2_ + x1, y2_ + y1)

                direction = np.array(pt2) - np.array(pt1)
                unit_vec = direction / np.linalg.norm(direction)
                length = max(h_img, w_img) * 2

                extended_pt1 = (
                    int(pt1[0] - unit_vec[0] * length),
                    int(pt1[1] - unit_vec[1] * length),
                )
                extended_pt2 = (
                    int(pt2[0] + unit_vec[0] * length),
                    int(pt2[1] + unit_vec[1] * length),
                )

                cv2.line(img0, extended_pt1, extended_pt2, (0, 255, 0), 2)
                bat_line_points = (extended_pt1, extended_pt2)

    return img0, current_hand_pos


def detect_webcam(
    target_class="bat", 
    weights="runs/train/final_learn3/weights/best.pt", 
    source=1, 
    top_crop=150,
    side_crop=200  # 양쪽에서 각각 잘라낼 픽셀 수
):
    model, stride, names, device = load_model(weights)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("⚠️ 웹캠을 열 수 없습니다.")
        return

    output_width, output_height = 1280, 900
    effective_height = output_height - top_crop  # 실제 표시될 높이
    effective_width = output_width - (2 * side_crop)  # 실제 표시될 너비

    print("컨트롤 방법:")
    print("  's': 인식 시작")
    print("  'e': 인식 중지")
    print("  'q': 프로그램 종료")

    detection_active = False
    initial_hand_pos = None
    threshold_exceeded = False
    MOVEMENT_THRESHOLD = 150

    # 새 변수들: 초기 기울기, 기울기 변경 감지, 타이머, 빨간색 상태
    initial_slope_sign = None
    slope_change_start = None
    SLOPE_MAINTAIN_DURATION = 0.01  # 기울기 유지 필요 시간
    red_triggered = False
    last_slope_status = None  # 마지막 기울기 상태 저장

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 웹캠에서 프레임을 읽을 수 없습니다.")
            break

        # 프레임 리사이즈 후 위쪽과 양옆 부분 잘라내기
        frame = cv2.resize(frame, (output_width, output_height))
        frame = frame[top_crop:, side_crop:output_width-side_crop]  # 위쪽과 양옆 부분 잘라내기
        display = frame.copy()

        status_text = "O" if detection_active else "X"
        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if detection_active else (0, 0, 255),
            2,
        )

        if detection_active:
            output, current_hand_pos = process_frame(frame, model, stride, names, device, target_class)

            # 1) 초기 손 위치 기록
            if initial_hand_pos is None and current_hand_pos is not None:
                initial_hand_pos = current_hand_pos
                print("초기 손 위치가 저장되었습니다:", initial_hand_pos)

            # 2) 손이 일정거리 이동하면 threshold_exceeded=True → 노란선 생성
            if initial_hand_pos is not None and current_hand_pos is not None and not threshold_exceeded:
                movement = current_hand_pos[0] - initial_hand_pos[0]
                if abs(movement) > MOVEMENT_THRESHOLD:
                    threshold_exceeded = True
                    print("임계값을 넘어섰습니다. 노란선을 생성합니다.")

                    # 노란선이 처음 나올 때, 배트 기울기 계산
                    if bat_line_points is not None:
                        pt1_init, pt2_init = bat_line_points
                        dy_init = pt2_init[1] - pt1_init[1]
                        dx_init = pt2_init[0] - pt1_init[0]
                        slope_init = dy_init / dx_init if dx_init != 0 else 0
                        initial_slope_sign = 1 if slope_init > 0 else -1
                        print("초기 배트 기울기 기호:", initial_slope_sign)

            # 3) 노란선 생성 이후: 노란선 그리기
            if threshold_exceeded and current_hand_pos is not None:
                yellow_line_x = current_hand_pos[0]
                cv2.line(output, (yellow_line_x, 0), (yellow_line_x, effective_height), (0, 255, 255), 2)

                # 배트 선분 존재 시 기울기 변경 감지
                if bat_line_points is not None and initial_slope_sign is not None:
                    pt1, pt2 = bat_line_points
                    dy = pt2[1] - pt1[1]
                    dx = pt2[0] - pt1[0]
                    slope = dy / dx if dx != 0 else 0

                    # 현재 기울기 상태 확인
                    current_slope_status = None
                    if initial_slope_sign > 0:  # 초기 기울기가 양수일 때
                        current_slope_status = slope <= 0
                    else:  # 초기 기울기가 음수일 때
                        current_slope_status = slope >= 0

                    # 기울기 상태가 변경되었을 때
                    if current_slope_status != last_slope_status:
                        if current_slope_status:  # 원하는 기울기 변화가 발생
                            slope_change_start = time.time()
                        else:  # 기울기가 원래대로 돌아감
                            slope_change_start = None
                    
                    # 기울기 상태 유지 시간 체크
                    if current_slope_status and slope_change_start is not None:
                        if time.time() - slope_change_start > SLOPE_MAINTAIN_DURATION and not red_triggered:
                            red_triggered = True
                            if initial_slope_sign > 0:
                                print("초기 기울기 양수 → 음수/0 변경 감지 및 유지. 빨간색 화면 ON.")
                            else:
                                print("초기 기울기 음수 → 양수/0 변경 감지 및 유지. 빨간색 화면 ON.")
                    
                    last_slope_status = current_slope_status

                # 빨간색 화면 표시
                if red_triggered:
                    overlay = np.zeros_like(output)
                    overlay[:] = (0, 0, 255)
                    alpha = 0.3
                    output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)
                    
                    # SWING 텍스트 추가
                    text = "SWING"
                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 2.0
                    thickness = 3
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    
                    # 텍스트 위치 계산 (오른쪽 위, 여백 20픽셀)
                    text_x = output.shape[1] - text_size[0] - 20
                    text_y = text_size[1] + 20
                    
                    # 텍스트 그리기 (테두리 효과를 위해 검은색으로 먼저 그리고 흰색으로 덮어씌우기)
                    cv2.putText(output, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
                    cv2.putText(output, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        else:
            output = frame

        cv2.imshow("Webcam Detection", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            detection_active = True
            initial_hand_pos = None
            threshold_exceeded = False
            initial_slope_sign = None
            slope_change_start = None
            last_slope_status = None
            red_triggered = False
            print("인식이 시작되었습니다.")
        elif key == ord("e"):
            detection_active = False
            initial_hand_pos = None
            threshold_exceeded = False
            initial_slope_sign = None
            slope_change_start = None
            last_slope_status = None
            red_triggered = False
            print("인식이 중지되었습니다.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_webcam()
