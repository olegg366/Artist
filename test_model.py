import mediapipe as mp
from utilites import draw_landmarks_on_image
import numpy as np
import cv2
import tensorflow as tf

def get_landmarks(detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    res = []

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        res.append([(l.x + l.y) / 2 for l in hand_landmarks])

    return res

vid = cv2.VideoCapture(0)

model = tf.keras.models.load_model('mlmodels/best.hdf5', compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy')

model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

classes = {
    0 : 'open',
    1 : 'up',
    2 : 'ok',
    3 : 'klick'
}

data = []

model.predict(np.zeros((1, 25, 21)))

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        res, img = vid.read()

        if not res: print(0)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        detection = landmarker.detect(mp_image)

        if detection.hand_landmarks:
            lmks = get_landmarks(detection)
            for lmk in lmks: data.append(lmk)
            if len(data) == 25:
                pred = model.predict(np.array([data], dtype='float64'), verbose=0)
                print(classes[np.argmax(pred)])
                data = []
            else:
                dt = data + [data[-1]] * (25 - len(data))
                pred = model.predict(np.array([dt], dtype='float64'), verbose=0)
                print(classes[np.argmax(pred)])

        img, x, y = draw_landmarks_on_image(img, detection)
        cv2.imshow('img', img)

        cv2.waitKey(1)

vid.release()
