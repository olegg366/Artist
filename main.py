import cv2
import pyautogui as pg
import numpy as np

import torch
import mediapipe as mp
import tensorflow as tf
from numba import cuda
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

from skimage.transform import resize

from utilites import draw_landmarks_on_image, draw, get_landmarks, dist
from interface import App
from google_speech import recognize
from get_trajectory import draw_img

from time import sleep

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

pg.FAILSAFE = False
device = cuda.get_current_device()
    
def get_func_from_saved_model(saved_model_dir):
   saved_model_loaded = tf.saved_model.load(
       saved_model_dir, tags=[tag_constants.SERVING])
   graph_func = saved_model_loaded.signatures[
       signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
   return graph_func, saved_model_loaded

print('Setting up widget...')
app = App()
print('Successfully set up widget.')

if __name__ == '__main__':
    #камера
    vid = cv2.VideoCapture(0)

    #модель распознавания жестов
    print('Loading gesture recognizer...')
    trt_func, _ = get_func_from_saved_model('mlmodels/static_tftrt')
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
        base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
        running_mode=VisionRunningMode.VIDEO)
    print('Succesfully set up hand landmarker.')

    #время последних жестов
    t = {
        'paint' : -1,
        'clean' : -1,
        'start' : -1
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
        'end': 0,
        'drag': 0
    }
    
    last_cords = []

    #индикатор того, рисуем мы или нет 
    flag = False
    end = False
    timestamp = 0
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            res, img = vid.read()

            if not res: 
                print(0)
                continue
            
            detection = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), timestamp)
            if detection.hand_landmarks:
                lmks = get_landmarks(detection)
                x, y = lmks[0, 8, :2]
                x *= img.shape[1]
                y *= img.shape[0]
                last_cords.append([x, y])
                if len(last_cords) > 20:
                    last_cords.pop(0)
                if len(last_cords) < 6:
                    timestamp += 1
                    continue
                # print(dist(lmks[0, 4], lmks[0, 8]) / dist(lmks[0, 0], lmks[0, 8]))
                if dist(lmks[0, 4], lmks[0, 8]) / dist(lmks[0, 0], lmks[0, 8]) <= 0.2:
                    gt = 'Click'
                else:
                    inp = {'conv1d_4_input': tf.convert_to_tensor(lmks[:, :, :2])}
                    pred = trt_func(**inp)['dense_5']
                    gt = classes[np.argmax(pred[0])]
                # print(gt)
                flagn, t, cnt, end = draw(gt, t, cnt, flag, last_cords, end)
                if flagn != flag and not end:
                    app.change_status()
                flag = flagn
                if end:
                    flag = False
                    end = False
                    cnt = {
                        'clean': 0,
                        'end': 0,
                        'drag': 0
                    }
                    t = {
                        'paint' : 0,
                        'clean' : 0,
                        'start' : 0
                    }
                    img = app.image
                    del trt_func
                    tf.keras.backend.clear_session()
                    sleep(2)
                    print('Setting up stable diffusion...')
                    img.save('scribble.png')
                    prompt, rus = recognize(app)
                    app.print_text('Вы сказали: ' + rus)
                    app.change_status()
                    app.update()
                    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", 
                                                                torch_dtype=torch.float32).to('cuda')   
                    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                                                            controlnet=controlnet, 
                                                                            safety_checker=None, 
                                                                            torch_dtype=torch.float32).to('cuda')
                    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
                    # pipe.enable_xformers_memory_efficient_attention()
                    # pipe.enable_model_cpu_offload()
                    print('Succesfully set up stable diffusion.')
                    img = pipe(prompt + ", beautiful", 
                        img, 
                        # guess_mode=True,
                        # guidance_scale=3.0,
                        negative_prompt="bad anatomy, low quality, worst quality", 
                        num_inference_steps=50, 
                        height=512, width=512).images[0]
                    del pipe
                    device.reset()
                    img.save('now.png')
                    app.display(img)
                    # draw_img(img)
                    app.change_status()
                    app.update()
                    try:
                        sleep(6000)
                    except KeyboardInterrupt:
                        app.remove_img()
                        # app.update()
            # cv2.imshow('img', img)
            img = draw_landmarks_on_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), detection)
            app.update((resize(img, (img.shape[0] // 2, img.shape[1] // 2)) * 255).astype('uint8'))
            cv2.waitKey(1)
            timestamp += 1

    vid.release()
