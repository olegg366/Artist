import mediapipe as mp
from utilites import draw_landmarks_on_image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.compiler.tensorrt import trt_convert as trt

def get_func_from_saved_model(saved_model_dir):
   saved_model_loaded = tf.saved_model.load(
       saved_model_dir, tags=[tag_constants.SERVING])
   graph_func = saved_model_loaded.signatures[
       signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
   return graph_func, saved_model_loaded

def get_landmarks(detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    res = []

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        res.append([[l.x, l.y] for l in hand_landmarks])

    return res

vid = cv2.VideoCapture(2)

trt_func, _ = get_func_from_saved_model('mlmodels/static_tftrt')

model_path = 'mlmodels/hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.GPU),
    running_mode=VisionRunningMode.IMAGE)

classes = {
    0 : 'Open_Palm',
    1 : 'Pointing_Up',
    2 : 'Thumb_Up'
}

print(trt_func(**{'conv1d_4_input': tf.zeros((1, 21, 2), dtype='float32')}))

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        res, img = vid.read()

        if not res: print(0)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        detection = landmarker.detect(mp_image)

        if detection.hand_landmarks:
            lmks = get_landmarks(detection)
            inp = {'conv1d_4_input': tf.convert_to_tensor(lmks)}
            pred = trt_func(**inp)['dense_5']
            print(classes[np.argmax(pred)])
            data = []

        img, x, y = draw_landmarks_on_image(img, detection)
        cv2.imshow('img', img)

        cv2.waitKey(1)

vid.release()
