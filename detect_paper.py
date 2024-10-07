import cv2
import numpy as np
from skimage.feature import canny
from skimage.measure import regionprops, label
from skimage.morphology import binary_dilation, square

def reorder_quadrilateral_vertices(vertices):
    center = np.mean(vertices, axis=0)
    
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    
    return vertices[sorted_indices]

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
 
    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)
 
    return cnt_scaled

def detect_paper(frame, warp=False):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_black = (0, 0, 0)
    upper_black = (180, 255, 100) 
    lower_green = np.array([30, 50, 50])
    upper_green = np.array([90, 255, 255])

    mask1 = cv2.inRange(hsv, lower_green, upper_green)
    mask2 = cv2.inRange(hsv, lower_black, upper_black)
    mask = mask1 & (~mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = np.array([])
    for idx, contour in enumerate(contours):
        if cv2.contourArea(contour) > 500:  
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
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
        w, h = 600, 725
        pts2 = np.float32([[0, 0], [0, w], [h, w], [h, 0]])
        matrix = cv2.getPerspectiveTransform(edges, pts2)
        roi = cv2.warpPerspective(frame, matrix, (h + 1, w + 1))
        frame = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower_green, upper_green)
        edges = canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) & (~mask)
        lb = label(edges)
        rps = regionprops(lb)
        if rps:
            areas = [rp.area for rp in rps]
            mx = areas.index(max(areas))
            rect = lb == (mx + 1)
            nz = np.array(list(zip(*np.nonzero(rect))))
            pnts = np.int64(cv2.boxPoints(cv2.minAreaRect(nz)))
            if pnts.size:
                print(pnts)
                pnts = scale_contour(pnts, 0.8)
                cv2.drawContours(frame, [pnts[:, [1, 0]]], 0, (0, 255, 0), 1)
                if warp:
                    return pnts, frame
                else:
                    return pnts
    if warp:
        return None, frame
    else:
        return None

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('Не удалось открыть камеру')
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ret, frame = cap.read()
        # frame = increase_brightness(frame, 50)
        if not ret:
            print("Не удалось прочитать кадр с камеры")
            break

        points, frame = detect_paper(frame, True)
        cv2.imshow('Detected Green Squares bbox', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
