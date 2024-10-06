import cv2
import numpy as np

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось прочитать кадр с камеры")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 60, 60])
    upper_green = np.array([80, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Минимальная площадь, задается эмпирически
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

    cv2.imshow('Detected Green Squares', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
