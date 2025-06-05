import cv2
import mediapipe as mp
import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
)


def load_model(weights="yolov5s.pt", device=""):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride = model.stride
    names = model.names
    return model, stride, names, device


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


def draw_bat_line(img0, det, names):
    annotator = Annotator(img0.copy())

    bat_head = None
    bat_grip = None

    # Extract detection boxes
    for *xyxy, conf, cls in reversed(det):
        c = int(cls)
        if names[c] == "bat_head":
            bat_head = [(int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)), conf]
        elif names[c] == "bat_grip":
            bat_grip = [(int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)), conf]

        # Draw boxes
        label = f"{names[c]} {conf:.2f}"
        annotator.box_label(xyxy, label, color=colors(c, True))

    # Draw line between head and grip if both are detected
    if bat_head and bat_grip:
        cv2.line(annotator.im, bat_head[0], bat_grip[0], (0, 255, 0), 2)

    return annotator.result(), bat_grip  # Return grip position for hand distance calculation


def process_hands(image):
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image_rgb)

    hand_landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks_single in results.multi_hand_landmarks:
            # Get wrist position
            wrist = (
                int(hand_landmarks_single.landmark[mp_hands.HandLandmark.WRIST].x * image.shape[1]),
                int(hand_landmarks_single.landmark[mp_hands.HandLandmark.WRIST].y * image.shape[0]),
            )
            hand_landmarks.append(wrist)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks_single,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
            )

    return image, hand_landmarks


def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def main(weights="best.pt", source="0", device=""):
    # Load model
    model, stride, names, device = load_model(weights, device)

    # Video capture
    if source.isnumeric():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process bat detection
        pred = process_image(model, frame, stride, device)

        bat_grip_pos = None
        # Process predictions
        for det in pred:
            if len(det):
                # Scale boxes to original image
                det[:, :4] = scale_coords(model.model.stride, det[:, :4], frame.shape).round()

                # Draw results
                frame, bat_grip_pos = draw_bat_line(frame, det, names)

        # Process hands detection
        frame, hand_positions = process_hands(frame)

        # Calculate and display distance between hands and bat grip
        if bat_grip_pos and hand_positions:
            for i, hand_pos in enumerate(hand_positions):
                distance = calculate_distance(hand_pos, bat_grip_pos[0])
                cv2.putText(
                    frame,
                    f"Hand {i + 1} to grip: {distance:.1f}px",
                    (10, 30 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

        # Show results
        cv2.imshow("Bat and Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
