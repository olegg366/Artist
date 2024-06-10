import mediapipe as mp
from utilites import draw_landmarks_on_image, get_landmarks, dist
import numpy as np
import cv2
import matplotlib.pyplot as plt

vid = cv2.VideoCapture(0)

model_path = 'mlmodels/hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.GPU),
    running_mode=VisionRunningMode.IMAGE)

ans = []
ds = []
mxlen = 25
try:
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            res, img = vid.read()
            if not res: 
                print('end')
                break
            
            # if img.mean() == 0:
            #     print(0)
            #     if len(ans):
            #         ds.append(ans)
            #         mxlen = max(mxlen, len(ans))
            #         ans = []
            #         continue

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            detection = landmarker.detect(mp_image)

            img = draw_landmarks_on_image(img, detection)
            cv2.imshow('img', img)

            if detection.hand_landmarks:
                res = get_landmarks(detection)[0]
                ans.append(dist(res[4], res[8]))
            cv2.waitKey(1)
except KeyboardInterrupt:
    pass
# for i in range(len(ds)):
#     if len(ds[i]) < mxlen:
#         ln = len(ds[i]) - 1
#         for j in range(mxlen - len(ds[i])):
#             ds[i].append(ds[i][ln])
# print(ans)
# f, ax = plt.subplots()
plt.rcParams.update({
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'xtick.labelbottom': True,
    'xtick.bottom': True,
    'ytick.labelleft': True,
    'ytick.left': True,
    'xtick.labeltop': True,
    'xtick.top': True,
    'ytick.labelright': True,
    'ytick.right': True
})
plt.plot(ans, linewidth=2)

# ax.set(xlabel='time (s)', ylabel='distance',
#        title='Distance between big and pointing fingers')
# ax.grid()
plt.show()
# ds = np.array(ds, dtype='float64')
# np.save('dataset/click.npy', ds)

vid.release()
