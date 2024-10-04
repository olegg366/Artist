import cv2
import pyautogui as pg
import numpy as np

import torch
import gc
import mediapipe as mp
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

# from DeepCache import DeepCacheSDHelper
from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler, ControlNetModel

from PIL import Image
from skimage.transform import resize
from skimage.filters import threshold_otsu

from utilites import draw_landmarks_on_image, draw, get_landmarks, dist
from interface import App
from google_speech import recognize
from get_trajectory import draw_img
from draw_logo import draw_a5

from time import sleep, time
import os
from random import choice

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

pg.FAILSAFE = False
    
def get_func_from_saved_model(saved_model_dir):
   saved_model_loaded = tf.saved_model.load(
       saved_model_dir, tags=[tag_constants.SERVING])
   graph_func = saved_model_loaded.signatures[
       signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
   return graph_func, saved_model_loaded

def run_app():
    global app
    while True:
        app.update()

def callback(_, step_index, timestep, callback_kwargs):
    app.progressbar_step(1)
    app.update()
    return callback_kwargs

def generate(img, prompt):
    image = np.array(img)
    c = image[:, :, 0]
    t = threshold_otsu(c)
    img = Image.fromarray((image <= t).astype('uint8') * 255)
    print('Setting up stable diffusion...')
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", 
                                             torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                                            controlnet=controlnet,
                                                            safety_checker=None, 
                                                            use_safetensors=True,
                                                            torch_dtype=torch.float32)

    # helper = DeepCacheSDHelper(pipe=pipe)
    # helper.set_params(
    #     cache_interval=5,
    #     cache_branch_id=0,
    # )
    # helper.enable()

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # pipe.enable_sequential_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)
    
    print('Succesfully set up stable diffusion.')

    image = pipe(prompt, img, 
                 num_inference_steps=50, 
                 negative_prompt="many lines, bad anatomy, worst quality, bad quality",
                 height=512, width=512, 
                 callback_on_step_end=callback).images[0]
    
    del pipe, controlnet
    gc.collect()
    torch.cuda.empty_cache()
    
    return image

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
        base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.GPU),
        num_hands=2,
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
    flag_checking = False
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
                if dist(lmks[0, 4], lmks[0, 8]) / dist(lmks[0, 0], lmks[0, 8]) <= 0.2:
                    gts = ['Click']
                else:
                    inp = {'conv1d_4_input': tf.convert_to_tensor(lmks[:, :, :2])}
                    pred = trt_func(**inp)['dense_5']
                    gts = [classes[x] for x in np.argmax(pred, axis=-1)]
                if flag_checking:
                    _, t, cnt, __ = draw(gts, t, cnt, True, last_cords, end, app)
                    if not app.flag_answer:
                        img = draw_landmarks_on_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), detection)
                        app.update((resize(img, (img.shape[0] // 2, img.shape[1] // 2)) * 255).astype('uint8'))
                        cv2.waitKey(1)
                        timestamp += 1
                        continue
                    else:
                        flag_checking = False
                else:
                    flagn, t, cnt, end = draw(gts, t, cnt, flag, last_cords, end, app)
                if flagn != flag and not end:
                    app.change_status()
                flag = flagn
                if end or app.flag_generate:
                    if not app.flag_answer:
                        scribble = app.image
                        scribble.save('images/scribble.png')
                        prompt = ''
                        while not prompt:
                            try:
                                prompt, rus = 'house', 'дом'
                            except ValueError:
                                app.print_text("Распознавание не удалось. Попробуйте ещё раз.")
                                sleep(3)
                                
                        app.print_text('Вы сказали: ' + rus + '?')
                        app.check_recognition()
                        flag_checking = 1
                        app.update()
                    else:
                        app.flag_answer = 0
                        flag_checking = 0
                        
                        if not app.flag_recognition:
                            timestamp += 1
                            app.update()
                            continue
                        app.print_text('Генерация по запросу: ' + rus)
                        app.change_status()
                        app.setup_progressbar()
                        
                        try:
                            gen = generate(scribble, prompt + ', russian, single color, easy drawing')
                        except:
                            ld = os.listdir('images/norm/')
                            file = ''
                            for fn in ld:
                                if fn.find(prompt) != -1:
                                    file = fn
                                    break
                            if not file:
                                gen = Image.open('images/norm/' + choice(ld))
                            else:
                                gen = Image.open('images/norm/' + file)
                        app.flag_generate = 0
                        
                        app.print_text('')
                        app.delete()
                        app.fr_progressbar.pack_forget()
                        
                        gen.save('images/generated/' + prompt.lower().replace(' ', '_') + '_' + str(time()) + '.png')
                        
                        app.display(gen)
                        
                        sleep(3)
                        draw_img(gen, k=512/370)
                        
                        app.change_status()                   
                        
                        flag = False
                        flagn = False
                        end = False
                        app.flag_answer = 0
                        
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
            img = draw_landmarks_on_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), detection)
            app.update((resize(img, (img.shape[0] // 2, img.shape[1] // 2)) * 255).astype('uint8'))
            cv2.waitKey(1)
            timestamp += 1

    vid.release()
