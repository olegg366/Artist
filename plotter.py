import os
import serial
from time import sleep
import mediapipe as mp
from math import cos, acos, pi
import cv2
from multiprocessing import Process, Value
from keras.models import load_model
from gesture_recognizer import BaseOptions, HandLandmarker, HandLandmarkerOptions, VisionRunningMode

class HandChecker:
    def __init__(
        self, 
        video_id, 
        pause, 
        landmarker_path = 'mlmodels/hand_landmarker.task',
        running_mode = 'VIDEO'
    ):
        self.video_id = video_id
        self.pause = pause
        self.landmarker_path = landmarker_path
        self.running_mode = running_mode
    
    def start_loop(self):
        self.terminate_flag = False
        self.process = Process(target=self.loop, daemon=True)
        self.process.start()
    
    def terminate(self):
        self.terminate_flag = True
        if self.process.is_alive():
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
    
    def loop(self):
        self.video = cv2.VideoCapture(self.video_id)
        timestamp = 0
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.landmarker_path, delegate=1),
            num_hands=2,
            running_mode=getattr(VisionRunningMode, self.running_mode)
        )
        
        self.landmarker = HandLandmarker.create_from_options(options)
        while not self.terminate_flag:
            flag, img = self.video.read()
            
            if not flag:
                print("Can't read image")
                continue
            
            timestamp += 1
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            detection = self.landmarker.detect_for_video(mediapipe_image, timestamp)
            
            self.pause.value = int(bool(detection.hand_landmarks))
            

def calculate_distance(ax, ay, bx, by):
    """
    Вычисляет расстояние между двумя точками.
    
    :param ax: Координата X первой точки.
    :param ay: Координата Y первой точки.
    :param bx: Координата X второй точки.
    :param by: Координата Y второй точки.
    :return: Расстояние между точками.
    """
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        

class Plotter(serial.Serial):
    def __init__(self, video_id, stop, speed = 8000, *args, **kwargs):
        port = kwargs['port']
        os.system(f'echo Dragon2009 | sudo -S chmod 666 {port}')
        super(Plotter, self).__init__(*args, **kwargs)
        sleep(2)
        self.write_command('G90')
        
        self.stop = stop
        
        self.video_id = video_id
        self.pause = Value('i', 0)
        self.down = 100
        self.speed = speed
        self.prev_angle = 0
        
    def write_command(self, command: str):
        command += '\n'
        self.write(command.encode())
        return self.read_until(b'ok\n').decode()
    
    def servo(self, angle):
        self.write_command('M400')
        return self.write_command(f'M280 P0 S{angle}')
    
    def get_desired_angle(angle):
        angle = angle / 180 * pi
        desired = acos((cos(angle) * 15 - 18) / 15)
        return desired / pi * 180
        
    def calibrate_servo(self):
        angle = 0
        self.servo(0)
        self.move_to(500, 0, 16000)
        while True:
            self.servo(angle)
            angle += 5
            status = self.write_command('M119')
            state = status[status.rfind('x_min: ') + 7:status.find('\n', status.rfind('x_min: '))]
            if state == 'TRIGGERED' or angle == 180:
                break
        desired = Plotter.get_desired_angle(angle)
        self.down = desired
        self.servo(0)
        
    def control_servo(self, command: str):
        """
        Управляет сервомотором на основе команды.
        
        :param ser: Объект Serial для связи с устройством.
        :param command: Команда ('up', 'maxup', 'down').
        """
        angle = 0
        if command == 'up': angle = 90
        elif command == 'maxup': angle = 0
        elif command == 'down': angle = self.down
        ret = self.servo(angle)
        sleep(abs(self.prev_angle - angle) * 0.002)
        self.prev_angle = angle
        return ret
    
    def generate_gcode(self, trajectory: list):
        """
        Генерирует G-код на основе траектории движения.
        
        :param trajectory: Список координат траектории.
        :return: Список команд G-кода.
        """
        i = 2
        gcode_commands = [(trajectory[0][0], trajectory[0][1], self.speed * 2), 'down']
        while i < len(trajectory):
            while i < len(trajectory) and trajectory[i][0] != 1e9:
                if abs(trajectory[i][0]) != 1e9:
                    gcode_commands += [(trajectory[i][0], trajectory[i][1], self.speed)]
                i += 1
            gcode_commands += ['up']
            if i < len(trajectory) - 1 and abs(trajectory[i + 1][0]) != 1e9:
                gcode_commands += [(trajectory[i + 1][0], trajectory[i + 1][1], self.speed * 2)]
            gcode_commands += ['down']
            i += 3
        return gcode_commands

    def move_to(self, x, y, speed = None):
        if speed is None: speed = self.speed
        self.write_command(f'G1 X{x} Y{y} F{speed}')
    
    def send_gcode(self, gcode_commands: list):
        """
        Отправляет G-код на устройство через последовательный порт.
        
        :param gcode_commands: Список команд G-кода.
        """
        hand_checker = HandChecker(self.video_id, self.pause)
        hand_checker.start_loop()
        
        prev_x, prev_y = 0, 0
        timestamp = 0
        i = 0
        try:
            while i < len(gcode_commands):
                timestamp += 1
                
                if self.pause.value: continue
                if self.stop.value:
                    self.stop.value = 0
                    break
                
                gcode = gcode_commands[i]
                i += 1
                # print(gcode)
                if isinstance(gcode, str):
                    self.control_servo(gcode)
                    continue
                if gcode[:2] == (prev_x, prev_y): 
                    continue
                self.move_to(*gcode)
                prev_x, prev_y = gcode[:2]
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(e)
        self.control_servo('maxup')
        self.move_to(0, 0, self.speed * 2)
    
    def plot_trajectory(self, trajectory):
        gcode = self.generate_gcode(trajectory)
        self.send_gcode(gcode)
        
    def interface(self):
        try:
            while True:
                command = input('Enter your command: ').strip()
                if command in ['up', 'down', 'maxup']:
                    print(self.control_servo(command))
                elif command[0] == 'X' or command[0] == 'Y':
                    print(self.write_command('G1 ' + command))
                else:
                    print(self.write_command(command))       
        except KeyboardInterrupt:
            self.control_servo('maxup')
            self.move_to(0, 0)
            
if __name__ == '__main__':
    plotter = Plotter(2, Value('i', 0), port='/dev/ttyACM0', baudrate=115200)
    plotter.interface()