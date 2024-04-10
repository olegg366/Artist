import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utilites import display_gesture

base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
  res, img = vid.read()
  if not res: continue

  cv2.imshow('img', img)
  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  recognition_result = recognizer.recognize(mp_image)
  if recognition_result.gestures:
    print(recognition_result.gestures)
    top_gesture = recognition_result.gestures[0][0]
    hand_landmarks = recognition_result.hand_landmarks

    display_gesture(mp_image, (top_gesture, hand_landmarks))

  cv2.waitKey(1)