import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def reorder_quadrilateral_vertices(vertices):
    center = np.mean(vertices, axis=0)
    
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    
    return vertices[sorted_indices]

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
    points = np.array([])
    for idx, contour in enumerate(contours):
        if cv2.contourArea(contour) > 1000:  
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
            if not points.size: points = approx[:, 0]
            else: points = np.append(points, approx[:, 0], axis=0)
    if len(points) >= 16:
        bp = cv2.boxPoints(cv2.minAreaRect(points))
        edges = []
        for edge in bp:
            md = 1e9
            point = None
            for pnt in points:
                if np.linalg.norm(edge - pnt) < md:
                    point = pnt
                    md = np.linalg.norm(edge - pnt)
            edges.append(point.tolist())
        edges = reorder_quadrilateral_vertices(np.array(edges, dtype='float32'))
        pts2 = np.float32([[0, 0], [0, 297], [210, 297], [210, 0]])
        matrix = cv2.getPerspectiveTransform(edges, pts2)
        frame = cv2.warpPerspective(frame, matrix, (210, 297))
        
        lower_black = (0, 0, 0)
        upper_black = (360, 255, 150) 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   
        mask = cv2.inRange(hsv, lower_black, upper_black)
        mask2 = cv2.inRange(hsv, lower_green, upper_green)
        mask = mask & (~mask2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 100:  
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 1)
        
    cv2.imshow('Detected Green Squares bbox', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
