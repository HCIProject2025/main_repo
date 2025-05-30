import cv2
import torch
import numpy as np
import mediapipe as mp
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def load_models(bat_weights, general_weights, device=''):
    device = select_device(device)
    bat_model = DetectMultiBackend(bat_weights, device=device)
    general_model = DetectMultiBackend(general_weights, device=device)
    return bat_model, general_model, bat_model.stride, bat_model.names, device


def process_image(model, img0, stride, device, conf_thres=0.25, iou_thres=0.45):
    # Padded resize
    img = letterbox(img0, stride=stride)[0]
    
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if len(img.shape) == 3:
        img = img[None]
    
    # Inference
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    return pred

def draw_detections(img0, det, names):
    annotator = Annotator(img0.copy())

    for *xyxy, conf, cls in reversed(det):
        c = int(cls)
        label = f'{names[c]} {conf:.2f}'
        annotator.box_label(xyxy, label, color=colors(c, True))

        if names[c].lower() == 'bat':
            x1, y1, x2, y2 = map(int, xyxy)

            # 중심 기준으로 바운딩 박스 축소
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1

            shrink_ratio = 0.6
            new_w = int(width * shrink_ratio)
            new_h = int(height * shrink_ratio)

            x1 = max(0, cx - new_w // 2)
            y1 = max(0, cy - new_h // 2)
            x2 = min(img0.shape[1], cx + new_w // 2)
            y2 = min(img0.shape[0], cy + new_h // 2)

            roi = img0[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            # 윤곽선 추출 - Canny + Morphology
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                box = np.intp(box)

                (cx, cy), (w, h), angle = rect

                if w > h:
                    dx, dy = (w / 2) * np.cos(np.deg2rad(angle)), (w / 2) * np.sin(np.deg2rad(angle))
                else:
                    dx, dy = (h / 2) * np.sin(np.deg2rad(angle)), (h / 2) * -np.cos(np.deg2rad(angle))

                pt1 = (int(cx + dx), int(cy + dy))
                pt2 = (int(cx - dx), int(cy - dy))

                head_local = pt1
                head_global = (head_local[0] + x1, head_local[1] + y1)
                cv2.circle(annotator.im, head_global, 8, (0, 255, 255), -1)

    return annotator.result()



def process_hands(image):
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect hands
    results = hands.process(image_rgb)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Add text to show it's a hand
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            x = int(wrist.x * image.shape[1])
            y = int(wrist.y * image.shape[0])
            cv2.putText(image, 'Hand', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return image

def main(bat_weights='runs/train/bat_detector3/weights/best.pt',
         general_weights='yolov5s.pt',
         source=1, device=''):

    # 두 모델 로드
    bat_model, general_model, stride, names, device = load_models(bat_weights, general_weights, device)

    if source.isnumeric():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 배트 감지
        bat_pred = process_image(bat_model, frame, stride, device)
        for det in bat_pred:
            if len(det):
                det[:, :4] = scale_boxes(bat_model.model.stride, det[:, :4], frame.shape).round()
                frame = draw_detections(frame, det, names)

        # 2. 일반 객체 감지 (예: 손이나 사람 등 필요 시 사용)
        # general_pred = process_image(general_model, frame, stride, device)
        # for det in general_pred:
        #     if len(det):
        #         det[:, :4] = scale_boxes(general_model.model.stride, det[:, :4], frame.shape).round()
        #         frame = draw_detections(frame, det, general_model.names)

        # 3. Mediapipe 손 추적
        frame = process_hands(frame)

        cv2.putText(frame, f'Press Q to quit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('YOLOv5 (bat+general) + Hand Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == '__main__':
    main(
        bat_weights='runs/train/bat_detector3/weights/best.pt',
        general_weights='yolov5s.pt',
        source='1'
    )
