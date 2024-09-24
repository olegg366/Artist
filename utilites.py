from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp

import os
import pyautogui as pg

from matplotlib import pyplot as plt
from time import time as tt
import numpy as np
import cv2

from interface import App

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'xtick.labelbottom': False,
    'xtick.bottom': False,
    'ytick.labelleft': False,
    'ytick.left': False,
    'xtick.labeltop': False,
    'xtick.top': False,
    'ytick.labelright': False,
    'ytick.right': False
})

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  x = None
  y = None
  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x * width for landmark in hand_landmarks]
    y_coordinates = [landmark.y * height for landmark in hand_landmarks]

    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def draw(tps: list, time, cnt, flag, cords, endflag, app: App):
    if 'Click' in tps and flag:
        cnt['clean'] = 0
        cnt['end'] = 0
        cnt['drag'] += 1
        x, y = cords[-5]
        x = 640 - x
        x = arduino_map(x, 0, 640, 0, 1920)
        y = arduino_map(y, 0, 480, 0, 1080)
        if flag and cnt['drag'] >= 3:
            pg.dragTo(x, y, 0.0, _pause=False)
        else:
            pg.moveTo(x, y, 0.0, _pause=False)
            pg.click()
    elif 'Pointing_Up' in tps or ('Click' in tps and not flag):
        x, y = cords[-1]
        x = 640 - x
        x = arduino_map(x, 0, 640, 0, 1920)
        y = arduino_map(y, 0, 480, 0, 1080)
        cnt['clean'] = 0
        cnt['end'] = 0
        cnt['drag'] = 0
        pg.moveTo(x, y, 0.0, _pause=False)
    elif flag and tps.count('Open_Palm') == 2:
        cnt['end'] = 0
        app.delete()
        time['clean'] = tt()
        cnt['clean'] = 0
    else:
        cnt['clean'] = 0
        if 'Thumb_Up' in tps and tt() - time['start'] > 10: 
            if cnt['end'] > 10:
                if not flag:
                    flag = True
                    cnt['end'] = 0
                    time['start'] = tt()
                    app.remove_instructions()
                    app.remove_img()
                else:
                    endflag = True
                    time['start'] = tt()
                    cnt['end'] = 0
                    flag = False
            else:
                cnt['end'] += 1
        else:
            cnt['end'] = 0
        
    return flag, time, cnt, endflag
  
def get_landmarks(detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    res = []

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        res.append([[l.x, l.y, l.z] for l in hand_landmarks])
    return np.array(res, dtype='float32')

def arduino_map(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5
