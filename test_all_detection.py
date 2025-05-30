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

def load_models(yolo_weights='yolov5s.pt', bat_weights='best.pt', device=''):
    # Load general YOLO model
    device = select_device(device)
    yolo_model = DetectMultiBackend(yolo_weights, device=device)
    yolo_stride = yolo_model.stride
    yolo_names = yolo_model.names

    # Load bat detection model
    bat_model = DetectMultiBackend(bat_weights, device=device)
    bat_stride = bat_model.stride
    bat_names = bat_model.names

    return (yolo_model, yolo_stride, yolo_names), (bat_model, bat_stride, bat_names), device

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

def draw_detections(img0, det, names, title=''):
    annotator = Annotator(img0.copy())
    
    # Draw boxes for YOLO detections
    for *xyxy, conf, cls in reversed(det):
        c = int(cls)
        label = f'{names[c]} {conf:.2f}'
        annotator.box_label(xyxy, label, color=colors(c, True))
    
    return annotator.result()

def draw_bat_line(img0, det, names):
    annotator = Annotator(img0.copy())
    
    bat_head = None
    bat_grip = None
    
    # Extract detection boxes
    for *xyxy, conf, cls in reversed(det):
        c = int(cls)
        if names[c] == 'bat_head':
            bat_head = [(int((xyxy[0] + xyxy[2])/2), int((xyxy[1] + xyxy[3])/2)), conf]
        elif names[c] == 'bat_grip':
            bat_grip = [(int((xyxy[0] + xyxy[2])/2), int((xyxy[1] + xyxy[3])/2)), conf]
        
        # Draw boxes
        label = f'{names[c]} {conf:.2f}'
        annotator.box_label(xyxy, label, color=colors(c, True))
    
    # Draw line between head and grip if both are detected
    if bat_head and bat_grip:
        cv2.line(annotator.im, bat_head[0], bat_grip[0], (0, 255, 0), 2)
    
    return annotator.result(), bat_grip

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
                int(hand_landmarks_single.landmark[mp_hands.HandLandmark.WRIST].y * image.shape[0])
            )
            hand_landmarks.append(wrist)
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks_single,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
    
    return image, hand_landmarks

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def main(yolo_weights='yolov5s.pt', bat_weights='best.pt', source='0', device=''):
    # Load models
    yolo_info, bat_info, device = load_models(yolo_weights, bat_weights, device)
    yolo_model, yolo_stride, yolo_names = yolo_info
    bat_model, bat_stride, bat_names = bat_info
    
    # Video capture
    if source.isnumeric():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process general YOLO detection
        yolo_pred = process_image(yolo_model, frame, yolo_stride, device)
        
        # Process bat detection
        bat_pred = process_image(bat_model, frame, bat_stride, device)
        
        # Process YOLO predictions
        for det in yolo_pred:
            if len(det):
                det[:, :4] = scale_boxes(yolo_model.model.stride, det[:, :4], frame.shape).round()
                frame = draw_detections(frame, det, yolo_names)
        
        # Process bat predictions
        bat_grip_pos = None
        for det in bat_pred:
            if len(det):
                det[:, :4] = scale_boxes(bat_model.model.stride, det[:, :4], frame.shape).round()
                frame, bat_grip_pos = draw_bat_line(frame, det, bat_names)
        
        # Process hands detection
        frame, hand_positions = process_hands(frame)
        
        # Calculate and display distance between hands and bat grip
        if bat_grip_pos and hand_positions:
            for i, hand_pos in enumerate(hand_positions):
                distance = calculate_distance(hand_pos, bat_grip_pos[0])
                cv2.putText(frame, f'Hand {i+1} to grip: {distance:.1f}px',
                           (10, 30 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show instructions
        cv2.putText(frame, f'Press Q to quit', (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show results
        cv2.imshow('All Detections', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == '__main__':
    # Use yolov5s.pt for general object detection and best.pt for bat detection
    main(yolo_weights='yolov5s.pt', bat_weights='best.pt', source='0') 