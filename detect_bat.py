import cv2
import torch
import numpy as np
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox

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
        # Add angle calculation if needed
        # angle = np.arctan2(bat_head[0][1] - bat_grip[0][1], bat_head[0][0] - bat_grip[0][0]) * 180 / np.pi
        # cv2.putText(annotator.im, f'Angle: {angle:.1f}°', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return annotator.result()

def main(weights='best.pt', source='0', device=''):
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
        
        # Process frame
        pred = process_image(model, frame, stride, device)
        
        # Process predictions
        for det in pred:
            if len(det):
                # Scale boxes to original image
                det[:, :4] = scale_coords(model.model.stride, det[:, :4], frame.shape).round()
                
                # Draw results
                frame = draw_bat_line(frame, det, names)
        
        # Show results
        cv2.imshow('Bat Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 