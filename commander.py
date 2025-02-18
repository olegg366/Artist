import sys
import signal
import numpy as np
import pyautogui as pg

from PIL import Image
from time import time, sleep

from multiprocessing import Process, Queue, Value, set_start_method

import matplotlib.pyplot as plt

from utilites import map_coords

from gesture_recognizer import GestureRecognizer
from generator import Generator
from interface import App
from drawer import Drawer
from string import punctuation

import speech_recognition as sr
from googletrans import Translator


pg.FAILSAFE = False


def remove_punctuation(s):
    res = ''
    for c in s.lower():
        if c not in punctuation:
            res += c
    return res.replace(' ', '_')


class Commander:
    def __init__(
        self, 
        frames_queue: Queue, 
        commands_queue: Queue, 
        images_queue: Queue,
        api_url: str,
        canvas_w, canvas_h, 
        shiftx, shifty,
        flag_recognition, flag_recognition_result,
        stop,
        flag_start,
        cpp_path: str, 
        lib_folder: str, 
        video_id: int,
        flag_end_plotting,
        cuda_version: str = '12.5', 
        gpu_arch: str = '86',
        port: str = '/dev/ttyACM0',
        baudrate: int = 115200,
    ):
        self.frames_queue = frames_queue
        self.commands_queue = commands_queue
        self.images_queue = images_queue
        
        self.api_url = api_url
        
        self.flag_recognition = flag_recognition
        self.flag_recognition_result = flag_recognition_result
        
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        
        self.shiftx = shiftx
        self.shifty = shifty
        
        self.stop = stop
        self.flag_start = flag_start
        self.cpp_path = cpp_path
        self.lib_folder = lib_folder
        self.video_id = video_id
        self.flag_end_plotting = flag_end_plotting
        self.cuda_version = cuda_version
        self.gpu_arch = gpu_arch
        self.port = port
        self.baudrate = baudrate
        
        self.reset_flags()

        
    def reset_flags(self):
        self.flag_recognition.value = 0
        self.flag_recognition_result.value = 0
        self.stop.value = 0
        self.flag_start.value = 0
        self.flag_end_plotting.value = 0
        self.flag_drawing = False
        self.flag_end = False
        self.flag_reset = True
        self.flag_drawing_line = False
        self.last_showed_end_time = -1
        self.gestures_in_row = {
            'clean': 0,
            'end': 0,
            'drag': 0
        }
    
    def move_while(self, cond: callable, check_gestures = False, delay = 3):
        tm = time()
        while cond():
            if self.frames_queue.empty():
                continue
            
            recognition_results = self.frames_queue.get()
            
            if recognition_results.landmarks is None:
                continue
            
            fx, fy = (recognition_results.landmarks[0, 8, :2] + recognition_results.landmarks[0, 4, :2]) / 2
            fx = recognition_results.image.shape[1] - fx * recognition_results.image.shape[1]
            fy *= recognition_results.image.shape[0]
            if check_gestures and (('Thumb_Up' in recognition_results.gestures and time() - self.last_showed_end_time > delay) or ('Thumb_Down' in recognition_results.gestures and time() - self.last_showed_end_time > delay)):
                self.last_showed_end_time = time()
                return 'Thumb_Up' in recognition_results.gestures
            self.move(recognition_results.gestures, fx, fy, recognition_results.image.shape[:2])
        return None
        
    def listen(self):
        recognizer = sr.Recognizer()
        translator = Translator()
        while not self.flag_recognition_result.value:
            self.flag_recognition_result.value = 0
            self.flag_recognition.value = 0
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                self.commands_queue.put(('print_text', ('Говорите...', )))
                audio = recognizer.listen(source, phrase_time_limit=5)
            try:
                self.commands_queue.put(('print_text', ('Идет распознавание...', )))
                text = recognizer.recognize_google(audio, language='ru-RU')
            except sr.exceptions.UnknownValueError as e:
                self.commands_queue.put(('print_text', ('Распознавание не удалось. Попробуйте ещё раз.', )))
                tm = time()
                self.move_while(lambda: time() - tm <= 2)
                continue
            self.commands_queue.put(('print_text', ('Идет перевод...', )))
            text_en = translator.translate(text, src='ru', dest='en').text
            # text, text_en = 'ананас', 'pineapple'
            print(text_en)
            self.commands_queue.put(('print_text', (f'Вы сказали: {text}?', )))
            self.commands_queue.put(('check_recognition', None))
            result = self.move_while(lambda: not self.flag_recognition.value, check_gestures=True)
            if result is not None:
                self.flag_recognition_result.value = int(result)
        return text, text_en
    
    def move(self, gestures, x, y, image_size):
        delta = 50
        w, h = pg.size()
        imgw, imgh = image_size
        
        xm = map_coords(x, 0, imgh, 0, w + delta / 2)
        ym = map_coords(y, 0, imgw, 0, h + delta / 2)
        
        if 'Click' in gestures: 
            pg.moveTo(xm, ym, 0.0, _pause=False)  
            if not self.flag_drawing_line:
                pg.click()
                self.flag_drawing_line = True
        elif 'Pointing_Up' in gestures:
            pg.moveTo(xm, ym, 0.0, _pause=False)
            self.flag_drawing_line = False
        else:
            self.flag_drawing_line = False
        
    def draw(self, gestures: list, x: int, y: int, image_size: tuple):
        imgw, imgh = image_size
        w, h = pg.size()
        x = imgh - x
        
        xc = map_coords(x, 0, imgh, -self.shiftx.value, self.canvas_w.value)
        yc = map_coords(y, 0, imgw, -self.shifty.value, self.canvas_h.value)
        self.move(gestures, x, y, image_size)
        if 'Click' in gestures and self.flag_drawing: 
            if xc > w * 0.2:   
                if not self.flag_drawing_line:      
                    self.commands_queue.put(('set_start', [(xc, yc)]))
                else:
                    self.commands_queue.put(('draw_line', [(xc, yc)]))
        elif 'Pointing_Up' in gestures or ('Click' in gestures and not self.flag_drawing):
            if xc > w * 0.2:
                self.commands_queue.put(('end_line', None))
        elif self.flag_drawing and gestures.count('Open_Palm') == 2:
            self.commands_queue.put(('delete', None))
        else:
            if 'Thumb_Up' in gestures and time() - self.last_showed_end_time > 3: 
                if not self.flag_drawing:
                    self.flag_drawing = True
                    self.flag_drawing_line = False
                    self.last_showed_end_time = time()
                    self.commands_queue.put(('remove_instructions', None))
                    self.commands_queue.put(('remove_img', None))
                    self.commands_queue.put(('change_status', None))
                else:
                    self.flag_end = True
                    self.last_showed_end_time = time()
                    self.flag_drawing = False
                    self.commands_queue.put(('change_status', None))
                    
        
    def loop(self):
        self.generation_queue = Queue()
        self.generator = Generator(self.generation_queue, self.commands_queue)
        
        self.drawer = Drawer(
            self.stop,
            self.flag_start,
            self.images_queue,
            self.cpp_path, 
            self.lib_folder, 
            self.video_id,
            self.flag_end_plotting,
            self.cuda_version, 
            self.gpu_arch,
            self.port,
            self.baudrate,
        )
        while not self.terminate_flag:
            if self.frames_queue.empty():
                continue
            
            recognition_results = self.frames_queue.get()
            
            if recognition_results.landmarks is None:
                continue
            
            fx, fy = (recognition_results.landmarks[0, 8, :2] + recognition_results.landmarks[0, 4, :2]) / 2
            fx *= recognition_results.image.shape[1]
            fy *= recognition_results.image.shape[0]
            self.draw(recognition_results.gestures, fx, fy, recognition_results.image.shape[:2])
            if self.flag_end:
                text_ru, text_en = self.listen()
                self.commands_queue.put(('delete_questions', None))
                self.generate(text_ru, text_en)
                self.plot(text_en)
                self.commands_queue.put(('change_status', None))
                self.commands_queue.put(('print_text', (f'', )))
                if self.flag_reset:
                    self.commands_queue.put(('reset_image', None))
                self.commands_queue.put(('remove_img', None))
                self.reset_flags()
                
    def generate(self, text_ru: str, text_en: str):
        self.commands_queue.put(('print_text', (f'Подождите, идёт генерация по запросу {text_ru}...', )))
        self.commands_queue.put(('setup_progressbar', None))
        self.commands_queue.put(('return_image', None))
        self.move_while(lambda: self.images_queue.empty())
        image = np.array(self.images_queue.get())
        
        self.generator.start_generation(image, text_en)
        self.move_while(lambda: self.generation_queue.empty())
        self.gen = self.generation_queue.get()

        self.commands_queue.put(('display', (self.gen, )))
        self.commands_queue.put(('remove_progressbar', None))
        self.commands_queue.put(('print_text', (f'Выберите изображение', )))
    
    def plot(self, prompt):
        self.move_while(lambda: self.images_queue.empty())
        self.commands_queue.put(('print_text', (f'Подождите, изображение обрабатывается...', )))
        image_idx = self.images_queue.get()
        image = self.gen[image_idx]
        image.save('images/generated/' + remove_punctuation(prompt) + '.png')
        
        self.commands_queue.put(('display_one', (image, )))
        self.drawer.start(np.array(image))
        
        self.move_while(lambda: self.images_queue.empty())
        self.commands_queue.put(('display_two', ([Image.fromarray((self.images_queue.get() * 255).astype('uint8')), image], )))
        self.commands_queue.put(('print_text', (f'Вам подходит то, что получится?', )))
        self.commands_queue.put(('check_recognition', None))
        self.flag_recognition.value = 0
        self.flag_recognition_result.value = 0
        res = self.move_while(lambda: not self.flag_recognition.value, check_gestures=True)
        self.commands_queue.put(('delete_questions', None))
        if res is not None:
            self.flag_recognition_result.value = res
        if not self.flag_recognition_result.value:
            self.flag_reset = False
            self.flag_start.value = 2
            return
        self.commands_queue.put(('print_text', (f'Пожалуйста, маркер в держатель и покажите большой палец.', )))
        res = self.move_while(lambda: True, check_gestures=True)
        self.flag_start.value = 1
        self.commands_queue.put(('print_text', (f'Подождите, пока ваше изображение нарисуется...', )))
        while True:
            self.flag_end_plotting.value = 0
            ret = self.move_while(lambda: not self.flag_end_plotting.value, check_gestures=True, delay=60)
            if ret is not None:
                if not ret:
                    self.stop.value = 1
                    break
            else: break
                
        self.commands_queue.put(('print_text', (f'Готово!', )))
        
        tm = time()
        self.move_while(lambda: time() - tm <= 3)
    
    def terminate(self):
        self.terminate_flag = True
        if self.process.is_alive():
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
        
    def start(self):
        self.terminate_flag = False
        self.process = Process(target=self.loop)
        self.process.start()
        
    def join(self):
        self.process.join()
        
if __name__ == '__main__':
    set_start_method('spawn')
    
    canvas_w = Value('i', 0)
    canvas_h = Value('i', 0)
    
    shiftx = Value('i', 0)
    shifty = Value('i', 0)
    
    flag_recognition = Value('i', 0)
    flag_recognition_result = Value('i', 0)
    
    stop = Value('i', 0)
    flag_start = Value('i', 0)
    
    frames_queue = Queue(-1)
    commands_queue = Queue(-1)
    images_queue = Queue(-1)
    
    cpp_path = 'trajectory.cpp'
    lib_folder = 'lib'
    second_video_id = 2
    flag_end_plotting = Value('i', 0)
    cuda_version = '11.8'
    gpu_arch = '86'
    port = '/dev/ttyACM0'
    baudrate = 115200
    
    api_url = 'https://qtf4vqzx-5000.euw.devtunnels.ms/generator'
    
    gesture_recognizer = GestureRecognizer(frames_queue, mode='tensorrt', recognizer_path='mlmodels/static_tftrt')
    app = App()
    com = Commander(
        frames_queue, 
        commands_queue, 
        images_queue,
        api_url,
        canvas_w, canvas_h, 
        shiftx, shifty, 
        flag_recognition, 
        flag_recognition_result,
        stop,
        flag_start,
        cpp_path, 
        lib_folder, 
        second_video_id,
        flag_end_plotting,
        cuda_version, 
        gpu_arch,
        port,
        baudrate,
    )

    
    gesture_recognizer.start_loop()
    com.start()
    
    def cleanup(signum, frame):
        print("Cleaning up resources...")
        com.terminate()
        app.running = False
        gesture_recognizer.terminate()
        frames_queue.close()
        commands_queue.close()
        frames_queue.join_thread()
        commands_queue.join_thread()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        app.mainloop(
            frames_queue, 
            commands_queue, 
            images_queue,
            canvas_w, canvas_h, 
            shiftx, shifty, 
            flag_recognition, flag_recognition_result
        )
    except KeyboardInterrupt:
        cleanup(None, None)
    gesture_recognizer.join()
    com.join()
        