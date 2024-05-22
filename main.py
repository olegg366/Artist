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

SERVO_DOWN = ''
SERVO_UP = ''

port = "COM1"  
baudrate = 112500 

def get_gcode(t: list):
    ans = 'G90'
    i = 1
    while i < len(t):
        ans += SERVO_DOWN
        i += 1
        while t[i] != 'up':
            ans += f'G1 X{t[i, 0]} Y{t[i, 1]}\n'
            i += 1
        ans += SERVO_UP
        if i < len(t) - 1:
            ans += f'G1 X{t[i + 1, 0]} Y{t[i + 1, 1]}'
    return ans

def draw(tp, time, flag, x, y, st):
    global app
    if tp == 'Pointing_Up' and flag:
        x = 640 - x
        x = arduino_map(x, 0, 640, 51, 1016)
        y = arduino_map(y, 0, 480, 237, 961)
        pg.dragTo(x, y)
        x -= app.canvas.winfo_x()
        y -= app.canvas.winfo_y() + 25
        x, y = map(int, (x, y))
        if st == 0:
            app.set_start(x, y)
            st = 1
        elif st == 1:
            app.draw_line(x, y)
        # else: pg.moveTo(x, y)
        time['paint'] = tt()
    elif tp == 'Open_Palm' and tt() - time['clean'] >= 2:
        pg.moveTo(47, 95)
        pg.click()

        pg.moveTo(2, 259)
        time['clean'] = tt()
    elif tp == 'Thumb_Up': 
        if not flag:
            flag = True
            time['start'] = tt()
        elif tt() - time['start'] > 10:
            pg.moveTo(716, 141)
            pg.click()
    return flag, time, st

def get_landmarks(detection_result, shp):
    hand_landmarks_list = detection_result.hand_landmarks
    res = []

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        res.append([[l.x * shp[1], l.y * shp[0]] for l in hand_landmarks])

    return np.array(res)

def gen_img(img: Image):
    img = pipe("mountains and lake, connect to contours, few details", 
           img, 
           negative_prompt = "many colours, many details", 
           num_inference_steps=10, 
           height=512, width=512).images[0]
    return img

def send_gcode(gcode: str):
    ser = serial.Serial(port, baudrate=baudrate)
    ser.write(gcode.encode())
    ser.close()

def draw_img(img: Image):
    img = np.array(img)
    img = get_colors(img)
    f = (img == 0).sum(axis=2) == 3
    f = ~f
    lb = label(~f)
    rgs = regionprops(lb)
    for rg in rgs:
        if rg.area < 30:
            f, img = fill(*rg.coords[0], f, img)

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
    
    gcode = get_gcode(all)
    send_gcode(gcode)

def run_app():
    global app
    while True:
        app.update()

print('Setting up widget...')
app = App(lambda img: gen_img(img))
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
    model_path = 'mlmodels\hand_landmarker.task'
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)
    print('Succesfully set up hand landmarker.')

    # настройка stable diffusion
    print('Setting up stable diffusion...')
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", 
                                                 torch_dtype=torch.float32).to('cuda')
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                                             controlnet=controlnet, 
                                                             safety_checker=None, 
                                                             torch_dtype=torch.float32).to('cuda')
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    print('Succesfully set up stable diffusion.')

    #перевод курсора в начальное положение
    pg.moveTo(2, 259)

    #время последних жестов
    t = {
        'paint' : time.time(),
        'clean' : time.time(),
        'start' : -1
    }

    #классы жестов
    classes = {
        0 : 'Open_Palm',
        1 : 'Pointing_Up',
        2 : 'Thumb_Up'
    }

    #индикатор того, рисуем мы или нет 
    flag = False

    #окно взаимодействия
    upd = Process(target=run_app)
    upd.start()

    st = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            try:
                res, img = vid.read()

                if not res: print(0)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                detection = landmarker.detect(mp_image)
                annotated_image, x, y = draw_landmarks_on_image(img, detection)
                cv2.imshow('img', annotated_image)
                if detection.hand_landmarks:
                    pred = model.predict(get_landmarks(detection, img.shape), verbose=0)
                    gt = classes[np.argmax(pred)]
                    # print(gt)
                    flag, t, st = draw(gt, t, flag, x, y, st)
                    print(app.line_points)

                # app.update()
                cv2.waitKey(1)
            except Exception as e:
                print(e)
                break

    upd.terminate()
    vid.release()