
import cv2


def resize_with_padding(image, target_size=(640, 640), color=(114, 114, 114)):
    h, w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded


# 이미지 불러오기
img_path = r"C:\incom\main_repo\bat_edge\image\image.png"
img = cv2.imread(img_path)
img_square = resize_with_padding(img, target_size=(640, 640))

# 전처리
gray = cv2.cvtColor(img_square, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# 윤곽선 추출
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img_square.copy()

# 조건에 맞는 contour만 추출
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = h / w if w != 0 else 0
    area = cv2.contourArea(cnt)

    # 배트의 조건: 세로로 길고 면적도 어느 정도 이상
    if aspect_ratio > 3 and area > 500:
        cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)

# 결과 시각화
cv2.imshow("Original", img_square)
cv2.imshow("Edges", edges)
cv2.imshow("Detected Bat", contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
