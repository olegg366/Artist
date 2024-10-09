import cv2
import pyautogui as pg
import numpy as np

import torch
import gc
import mediapipe as mp
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

from DeepCache import DeepCacheSDHelper
from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler, ControlNetModel

from PIL import Image
from skimage.transform import resize
from skimage.filters import threshold_otsu

from utilities import draw_landmarks_on_image, draw, get_landmarks, dist
from interface import App
from google_speech import recognize_and_translate
from get_trajectory import draw_img
from draw_logo import draw_a5

from time import sleep, time
import os
from random import choice

# Настройка GPU для TensorFlow
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

pg.FAILSAFE = False

def load_model_from_saved_dir(saved_model_dir):
    """
    Загружает модель из директории сохраненной модели.
    
    :param saved_model_dir: Путь к директории сохраненной модели.
    :return: Функция для выполнения модели и загруженная модель.
    """
    saved_model_loaded = tf.saved_model.load(
        saved_model_dir, tags=[tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    return graph_func, saved_model_loaded

def run_app():
    """
    Запускает основной цикл обновления приложения.
    """
    global app
    while True:
        app.update()

def callback(_, step_index, timestep, callback_kwargs):
    """
    Колбэк-функция для обновления прогресс-бара при генерации изображения.
    
    :param _: Не используется.
    :param step_index: Индекс текущего шага.
    :param timestep: Текущий временной шаг.
    :param callback_kwargs: Дополнительные аргументы.
    :return: Обновленные аргументы.
    """
    app.progressbar_step(1)
    app.update()
    return callback_kwargs

def generate_image(image, prompt):
    """
    Генерирует изображение с использованием Stable Diffusion.
    
    :param image: Исходное изображение.
    :param prompt: Текстовое описание для генерации.
    :return: Сгенерированное изображение.
    """
    image_array = np.array(image)
    channel = image_array[:, :, 0]
    threshold = threshold_otsu(channel)
    image = Image.fromarray((image_array <= threshold).astype('uint8') * 255)
    
    print('Настройка Stable Diffusion...')
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", 
                                                 torch_dtype=torch.float32).to('cuda')
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                                              controlnet=controlnet,
                                                              safety_checker=None, 
                                                              use_safetensors=True,
                                                              torch_dtype=torch.float32).to('cuda')

    helper = DeepCacheSDHelper(pipe=pipe)
    helper.set_params(
        cache_interval=5,
        cache_branch_id=0,
    )
    helper.enable()

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_xformers_memory_efficient_attention()

    pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)
    
    print('Stable Diffusion успешно настроен.')

    generated_image = pipe(prompt, image, 
                           num_inference_steps=50, 
                           negative_prompt="many lines, bad anatomy, worst quality, bad quality",
                           height=512, width=512, 
                           callback_on_step_end=callback).images[0]
    
    del pipe, controlnet
    gc.collect()
    torch.cuda.empty_cache()
    
    return generated_image

print('Настройка виджета...')
app = App()
print('Виджет успешно настроен.')

if __name__ == '__main__':
    # Инициализация камеры
    camera = cv2.VideoCapture(2)

    # Загрузка модели распознавания жестов
    print('Загрузка модели распознавания жестов...')
    gesture_recognizer_func, _ = load_model_from_saved_dir('mlmodels/static_tftrt')
    print('Модель распознавания жестов успешно загружена.')

    # Настройка разметки руки
    print('Настройка разметки руки...')
    hand_landmarker_model_path = 'mlmodels/hand_landmarker.task'
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hand_landmarker_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_landmarker_model_path, delegate=BaseOptions.Delegate.GPU),
        num_hands=2,
        running_mode=VisionRunningMode.VIDEO)
    print('Разметка руки успешно настроена.')

    # Время последних жестов
    last_gesture_times = {
        'paint' : -1,
        'clean' : -1,
        'start' : -1
    }

    # Классы жестов
    gesture_classes = {
        0 : 'Open_Palm',
        1 : 'Pointing_Up',
        2 : 'Thumb_Up'
    }
    
    # Счетчик жестов подряд
    gesture_count = {
        'clean': 0,
        'end': 0,
        'drag': 0
    }
    
    last_coordinates = []

    # Индикатор того, рисуем мы или нет 
    is_drawing = False
    is_drawing_new = False
    is_end = False
    is_checking = False
    timestamp = 0
    with HandLandmarker.create_from_options(hand_landmarker_options) as landmarker:
        while True:
            ret, frame = camera.read()

            if not ret: 
                print(0)
                continue
            
            detection = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), timestamp)
            if detection.hand_landmarks:
                landmarks = get_landmarks(detection)
                x, y = landmarks[0, 8, :2]
                x *= frame.shape[1]
                y *= frame.shape[0]
                last_coordinates.append([x, y])
                if len(last_coordinates) > 20:
                    last_coordinates.pop(0)
                if len(last_coordinates) < 6:
                    timestamp += 1
                    continue
                if dist(landmarks[0, 4], landmarks[0, 8]) / dist(landmarks[0, 0], landmarks[0, 8]) <= 0.2:
                    gestures = ['Click']
                else:
                    input_data = {'conv1d_4_input': tf.convert_to_tensor(landmarks[:, :, :2])}
                    predictions = gesture_recognizer_func(**input_data)['dense_5']
                    gestures = [gesture_classes[x] for x in np.argmax(predictions, axis=-1)]
                if is_checking:
                    _, last_gesture_times, gesture_count, __ = draw(gestures, last_gesture_times, gesture_count, True, last_coordinates, is_end, app)
                    if not app.flag_answer:
                        frame = draw_landmarks_on_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), detection)
                        app.update((resize(frame, (frame.shape[0] // 2, frame.shape[1] // 2)) * 255).astype('uint8'))
                        cv2.waitKey(1)
                        timestamp += 1
                        continue
                    else:
                        is_checking = False
                else:
                    is_drawing_new, last_gesture_times, gesture_count, is_end = draw(gestures, last_gesture_times, gesture_count, is_drawing, last_coordinates, is_end, app)
                if is_drawing_new != is_drawing and not is_end:
                    app.change_status()
                is_drawing = is_drawing_new
                if is_end or app.flag_generate:
                    if not app.flag_answer:
                        scribble = app.image
                        scribble.save('images/scribble.png')
                        prompt = ''
                        while not prompt:
                            try:
                                prompt, rus = recognize_and_translate(app)
                            except ValueError:
                                app.print_text("Распознавание не удалось. Попробуйте ещё раз.")
                                sleep(3)
                                
                        app.print_text('Вы сказали: ' + rus + '?')
                        frame = np.zeros_like(frame)
                        app.check_recognition()
                        is_checking = 1
                        app.update()
                    else:
                        app.flag_answer = 0
                        is_checking = 0
                        
                        # Проверяем, можно ли генерировать
                        if not app.flag_recognition:
                            timestamp += 1
                            app.update()
                            continue
                        
                        # Выводим информацию 
                        app.print_text('Генерация по запросу: ' + rus)
                        app.change_status()
                        app.setup_progressbar()
                        
                        # Генерируем изображение
                        try:
                            generated_image = generate_image(scribble, prompt + ', simple drawing')
                        except Exception as e:
                            print(e)
                            ld = os.listdir('images/norm/')
                            file = ''
                            for fn in ld:
                                if fn.find(prompt) != -1:
                                    file = fn
                                    break
                            if not file:
                                generated_image = Image.open('images/norm/' + choice(ld))
                            else:
                                generated_image = Image.open('images/norm/' + file)
                        app.flag_generate = 0
                        
                        # Удаляем прогрессбар
                        app.print_text('')
                        app.delete()
                        app.fr_progressbar.pack_forget()
                        
                        generated_image.save('images/generated/' + prompt.lower().replace(' ', '_') + '_' + str(time()) + '.png')
                        generated_image.save('images/gen.png')  
                        # Отображаем изображение
                        app.display(generated_image)
                        
                        sleep(3)
                        draw_img(generated_image)
                        # Меняем статус
                        app.change_status()                   
                        
                        # Обнуляем флаги
                        is_drawing = False
                        is_drawing_new = False
                        is_end = False
                        app.flag_answer = 0
                        
                        gesture_count = {
                            'clean': 0,
                            'end': 0,
                            'drag': 0
                        }
                        last_gesture_times = {
                            'paint' : 0,
                            'clean' : 0,
                            'start' : 0
                        }
            # Отрисовываем ключевые точки и обновляем интерфейс
            frame = draw_landmarks_on_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), detection)
            app.update((resize(frame, (frame.shape[0] // 2, frame.shape[1] // 2)) * 255).astype('uint8'))
            cv2.waitKey(1)
            timestamp += 1

    camera.release()