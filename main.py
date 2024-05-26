import os
os.system('CC=nvc python compile_cython.py build_ext --inplace')

import mediapipe as mp
import pyautogui as pg
import cv2
import numpy as np
import tensorflow as tf
import time
from skimage.morphology import label
from skimage.measure import regionprops
from multiprocessing import Process
from utilites import draw_landmarks_on_image, arduino_map
from interface import App
from time import time as tt
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from get_colors import get_colors
from accelerated_trajectory import fill, compute_image
import serial
from numba import cuda
from time import sleep
import pickle
device = cuda.get_current_device()

DOWN = 2000
UP = 1000

port = "/dev/ttyACM0"  
baudrate = 115200 

speed = 5000

def servo(ser, n):
    return None
    # cnt = 100
    # if n == UP:
    #     cnt = 1
    # for i in range(cnt):
    #     ser.write(b'M42 P12 S255 T1\n')
    #     sleep(n / 1000000)
    #     ser.write(b'M42 P12 S0 T1\n')

def get_gcode(t: list):
    i = 1
    k = 3
    ans = []
    while i < len(t):
        ans += ['down']
        i += 1
        while i < len(t) and t[i] != 'up':
            try:
                ans += [f'G1 X{t[i][0] / k} Y{t[i][1] / k} F{speed}\n']
            except TypeError:
                pass
            i += 1
        ans += ['up']
        if i < len(t) - 1:
            try:
                ans += [f'G1 X{t[i + 1][0] / k} Y{t[i + 1][1] / k} F{speed}\n']
            except TypeError:
                pass
    return ans

def draw(tp, time, cnt, flag, x, y, endflag):
    global app
    if tp == 'Pointing_Up' and flag:
        x = 640 - x
        x = arduino_map(x, 0, 640, 75, 1920)
        y = arduino_map(y, 0, 480, 65, 1080)
        cnt['clean'] = 0
        cnt['end'] = 0
        pg.dragTo(x, y)
    elif flag and tp == 'Open_Palm' and tt() - time['clean'] > 5:
        cnt['end'] = 0
        if cnt['clean'] > 20:
            pg.moveTo(155, 140)
            pg.click()

            pg.moveTo(75, 216)
            time['clean'] = tt()
            cnt['clean'] = 0
        else:
            cnt['clean'] += 1
    else:
        cnt['clean'] = 0
        if tp == 'Thumb_Up' and tt() - time['start'] > 10: 
            if cnt['end'] > 10:
                if not flag:
                    flag = True
                    cnt['end'] = 0
                    time['start'] = tt()
                else:
                    pg.moveTo(638, 138)
                    pg.click()
                    endflag = True
                    time['start'] = tt()
                    cnt['end'] = 0
                    flag = False
            else:
                cnt['end'] += 1
        else:
            cnt['end'] = 0
        
    return flag, time, cnt, endflag

def get_landmarks(detection_result, shp):
    hand_landmarks_list = detection_result.multi_hand_landmarks
    res = []

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx].landmark
        res.append([[l.x * shp[1], l.y * shp[0]] for l in hand_landmarks])

    return np.array(res)

def dist(ax, ay, bx, by):
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

def send_gcode(gcodes: list):
    ser = serial.Serial(port, baudrate=baudrate)
    sleep(2)
    ser.write(b'G90\n')
    prevx, prevy = 0, 0
    for gcode in gcodes:
        print(gcode)
        if gcode == 'up':
            servo(ser, UP)
            sleep(0.2)
        elif gcode == 'down':
            servo(ser, DOWN)
            sleep(0.2)
        else:
            if gcode == f"G1 X{prevx} Y{prevy} F{speed}": 
                continue
            ser.write(gcode.encode())
            gcode = [float(gcode[gcode.index('X') + 1:gcode.index('Y') - 1]), float(gcode[gcode.index('Y') + 1:gcode.index('F') - 1])]
            d = dist(prevx, prevy, gcode[0], gcode[1]) / (speed / 60)
            sleep(d + 0.2)
            prevx, prevy = gcode[0], gcode[1]
    ser.close()

def draw_img(img: Image):
    img = np.array(img)
    print('getting colors..')
    img = get_colors(img)
    print('got colors')
    f = (img == 0).sum(axis=2) == 3
    f = ~f
    lb = label(~f)
    rgs = regionprops(lb)
    for rg in rgs:
        if rg.area < 30:
            f, img = fill(*rg.coords[0], f, img)

    cv2.imshow('imgg', img)
    cv2.waitKey(1)
    sleep(5)
    print('getting trajectory...')
    clrs = np.unique(img.reshape(-1, img.shape[-1]), axis=0)
    idx = 0
    all = []
    for i in range(1, len(clrs)):
        clr = clrs[i]
        f = (img == clr).sum(axis=2) == 3
        lb = label(f)
        rgs = regionprops(lb)
        for reg in rgs:
            idx += 1
            regimg = reg.image
            cords = compute_image(regimg, 10, *reg.bbox[:2])
            all.append('down')
            all.extend(cords)
            all.append('up')
    print('got trajectory')
    
    with open('last_trajectory.lst', 'wb') as f:
        pickle.dump(all, f)
    
    print('sending gcode...')
    gcode = get_gcode(all)
    send_gcode(gcode)
    print('sent gcode')

def run_app():
    global app
    while True:
        app.update()

print('Setting up widget...')
app = App()
print('Successfully set up widget.')

if __name__ == '__main__':
    #камера
    vid = cv2.VideoCapture(0)

    #модель распознавания жестов
    print('Loading gesture recognizer...')
    model = tf.keras.models.load_model('mlmodels/static', compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.predict(np.zeros((1, 21, 2)), verbose=0)
    print('Succesfully loaded gesture recognizer.')

    #разметка руки
    print('Setting up hand landmarker...')
    model_path = 'mlmodels/hand_landmarker.task'
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)
    print('Succesfully set up hand landmarker.')

    # настройка stable diffusion
    #перевод курсора в начальное положение
    pg.moveTo(2, 259)

    #время последних жестов
    t = {
        'paint' : time.time(),
        'clean' : time.time(),
        'start' : time.time()
    }

    #классы жестов
    classes = {
        0 : 'Open_Palm',
        1 : 'Pointing_Up',
        2 : 'Thumb_Up'
    }
    
    # счетчик жестов подряд
    cnt = {
        'clean': 0,
        'end': 0
    }

    #индикатор того, рисуем мы или нет 
    flag = False

    #окно взаимодействия
    # upd = Process(target=run_app)
    # upd.start()

    end = False

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        max_num_hands=1,
        min_tracking_confidence=0.5) as hands:
        while True:
            res, img = vid.read()

            if not res: 
                print(0)
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detection = hands.process(img)
            if detection.multi_hand_landmarks:
                for hand_landmarks in detection.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                lmks = get_landmarks(detection, img.shape)
                x, y = lmks[0][8]
                try:
                    pred = model.predict(lmks, verbose=0)
                except Exception:
                    model = tf.keras.models.load_model('mlmodels/static', compile=False)
                    model.compile(optimizer='adam', loss='categorical_crossentropy')
                    pred = model.predict(lmks, verbose=0)
                gt = classes[np.argmax(pred[0])]
                flagn, t, cnt, end = draw(gt, t, cnt, flag, x, y, end)
                if flagn != flag:
                    app.change_status()
                flag = flagn
                if end:
                    flag = False
                    end = False
                    cnt = {
                        'clean': 0,
                        'end': 0
                    }
                    t = {
                        'paint' : 0,
                        'clean' : 0,
                        'start' : 0
                    }
                    img = app.image
                    del model
                    device.reset()
                    sleep(2)
                    print('Setting up stable diffusion...')
                    img.save('scribble.png')
                    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", 
                                                                torch_dtype=torch.float32).to('cuda')
                    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                                                            controlnet=controlnet, 
                                                                            safety_checker=None, 
                                                                            torch_dtype=torch.float32).to('cuda')
                    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
                    # pipe.enable_xformers_memory_efficient_attention()
                    pipe.enable_model_cpu_offload()
                    print('Succesfully set up stable diffusion.')
                    img = pipe("flower", 
                        img, 
                        # guess_mode=True,
                        # guidance_scale=3.0,
                        negative_prompt="bad anatomy, low quality, worst quality", 
                        num_inference_steps=10, 
                        height=320, width=320).images[0]
                    del pipe
                    device.reset()
                    img.save('now.png')
                    app.display(img)
                    # draw_img(img)
                    app.change_status()
                    app.update()
                    try:
                        sleep(60)
                    except KeyboardInterrupt:
                        app.remove()
                        # app.update()
            cv2.imshow('img', img)

            app.update()
            cv2.waitKey(1)

    # upd.terminate()
    vid.release()