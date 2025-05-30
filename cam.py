import cv2

print("Press 'q' to quit.")

for i in range(5):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"Camera index {i} not available.")
        continue

    print(f"Showing camera index {i}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(f'Camera {i}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()