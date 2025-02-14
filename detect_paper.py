import cv2
import numpy as np
from skimage.feature import canny
from skimage.measure import regionprops, label
from skimage.morphology import binary_dilation, square
from utilites import dist2

# Функция для переупорядочивания вершин четырехугольника по часовой стрелке
def reorder_quadrilateral_vertices(vertices):
    center = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return vertices[sorted_indices]

# Функция для увеличения яркости изображения
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

# Функция для масштабирования контура
def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)
    return cnt_scaled

# Основная функция для обнаружения бумаги и зеленых квадратов
def detect_paper(frame, warp=False):
    frame = frame[:, 200:950]
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Определяем диапазоны цветов для черной и зеленой маски
    lower_black = (0, 0, 0)
    upper_black = (180, 255, 100)
    lower_green = np.array([25, 20, 50])
    upper_green = np.array([80, 200, 200])

    # Создаем маски для зеленого и черного цветов
    mask1 = cv2.inRange(hsv, lower_green, upper_green)
    mask2 = cv2.inRange(hsv, lower_black, upper_black)
    mask = mask1 & (~mask2)
    # cv2.imshow('sus', mask)

    # Находим контуры на маске
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    points = np.array([])

    # Обрабатываем каждый контур
    for idx, contour in enumerate(contours):
        if cv2.contourArea(contour) > 800 and cv2.contourArea(contour) <= 5000:
            
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            bbox = cv2.boxPoints(cv2.minAreaRect(approx))
            w, h = sorted([dist2(bbox[0], bbox[1]), dist2(bbox[1], bbox[2])])
            if abs(w / h - 1) > 0.7:
                continue
            cv2.drawContours(frame, [cv2.boxPoints(cv2.minAreaRect(approx)).astype('int')], -1, (0, 0, 255), 5)
            if not points.size:
                points = approx[:, 0]
            else:
                points = np.append(points, approx[:, 0], axis=0)
    # cv2.imshow('Detected Green Squares bbox', frame)
    # Если найдено достаточно точек, обрабатываем их
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
        cv2.drawContours(frame, [edges.astype('int64')], -1, (255, 0, 0), 1)

        # Определяем размеры для перспективной трансформации
        w, h = 600, 725
        pts2 = np.float32([[0, 0], [0, w], [h, w], [h, 0]])
        matrix = cv2.getPerspectiveTransform(edges, pts2)
        roi = cv2.warpPerspective(frame, matrix, (h + 1, w + 1))
        roi = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Обрабатываем ROI для обнаружения зеленых квадратов
        mask = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), lower_green, upper_green)
        edges = canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)) & (~mask)
        lb = label(edges)
        rps = regionprops(lb)
        if rps:
            areas = [rp.area for rp in rps]
            mx = areas.index(max(areas))
            rect = lb == (mx + 1)
            nz = np.array(list(zip(*np.nonzero(rect))))
            pnts = np.int64(cv2.boxPoints(cv2.minAreaRect(nz)))
            if pnts.size:
                try:
                    pnts = scale_contour(pnts, 0.8)
                except ZeroDivisionError:
                    if warp:
                        return None, roi
                    else:
                        return None
                if warp:
                    return pnts, roi
                else:
                    return pnts
    if warp:
        return None, frame
    else:
        return None

# Основной блок программы
if __name__ == '__main__':
    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        print('Не удалось открыть камеру')
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось прочитать кадр с камеры")
            break

        points, frame = detect_paper(frame, True)
        cv2.imshow('Detected Green Squares bbox', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()