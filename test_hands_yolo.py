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

def load_model(weights='yolov5s.pt', device=''):
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

def draw_detections(img0, det, names):
    annotator = Annotator(img0.copy())
    
    # Draw boxes for YOLO detections
    for *xyxy, conf, cls in reversed(det):
        c = int(cls)
        label = f'{names[c]} {conf:.2f}'
        annotator.box_label(xyxy, label, color=colors(c, True))
    
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

def main(weights='yolov5s.pt', source='0', device=''):
    # Load YOLO model
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
        
        # Process YOLO detection
        pred = process_image(model, frame, stride, device)
        
        # Process predictions
        for det in pred:
            if len(det):
                # Scale boxes to original image
                det[:, :4] = scale_boxes(model.model.stride, det[:, :4], frame.shape).round()
                
                # Draw YOLO results
                frame = draw_detections(frame, det, names)
        
        # Process hands detection and draw landmarks
        frame = process_hands(frame)
        
        # Show FPS
        cv2.putText(frame, f'Press Q to quit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show results
        cv2.imshow('YOLOv5 + Hand Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == '__main__':
    # Use yolov5s.pt for general object detection
    main(weights='yolov5s.pt', source='0') 