import mediapipe as mp
from utilites import draw_landmarks_on_image
import numpy as np
import cv2

def get_landmarks(detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    res = []
    # h, w = shp[:2]

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        res.append([[l.x, l.y] for l in hand_landmarks])

    return res

vid = cv2.VideoCapture('D:/Oleg/VIdeos/no.mp4')

model_path = 'mlmodels/hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

ans = []
ds = []
mxlen = 25
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        res, img = vid.read()
        if not res: 
            print('end')
            break
        
        if len(ans) == 25:
            print(0)
            if len(ans):
                ds.append(ans)
                mxlen = max(mxlen, len(ans))
                ans = []
                continue

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        detection = landmarker.detect(mp_image)

        img = draw_landmarks_on_image(img, detection)
        # cv2.imshow('img', img)

        if detection.hand_landmarks:
            res = get_landmarks(detection)
            ans.append(res)
        # cv2.waitKey(1)
for i in range(len(ds)):
    if len(ds[i]) < mxlen:
        ln = len(ds[i]) - 1
        for j in range(mxlen - len(ds[i])):
            ds[i].append(ds[i][ln])
ds = np.array(ds, dtype='float64')
np.save('dataset/no.npy', ds)

vid.release()
