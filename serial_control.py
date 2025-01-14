import serial
from time import sleep
import mediapipe as mp
import numpy as np
from math import cos, acos, pi
import os
import cv2

# Устанавливаем права доступа к порту
os.system('echo Dragon2009 | sudo -S chmod 666 /dev/ttyACM0')

# Настройки порта и скорости передачи данных
port = "/dev/ttyACM0"  
baudrate = 115200 

# Скорость перемещения
speed = 8000

down = 135

# Настройки для MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='mlmodels/hand_landmarker.task', delegate=BaseOptions.Delegate.CPU),
    running_mode=VisionRunningMode.VIDEO)

def get_desired_angle(angle):
    angle = angle / 180 * pi
    desired = acos((cos(angle) * 15 - 18) / 15)
    return desired / pi * 180

def calibrate(ser: serial.Serial):
    global down
    angle = 0
    while True:
        ser.write(f'M280 P0 S{angle}\n'.encode())
        ser.read_until(b'ok\n')
        angle += 5
        ser.write(b'M119\n')
        status = ser.read_until(b'ok\n').decode()
        state = status[status.rfind('x_min: ') + 7:status.find('\n', status.rfind('x_min: '))]
        if state == 'TRIGGERED' or angle == 180:
            break
    desired = get_desired_angle(angle)
    down = desired
    ser.write(b'M280 P0 S0\n')
    

def control_servo(ser: serial.Serial, command: str):
    """
    Управляет сервомотором на основе команды.
    
    :param ser: Объект Serial для связи с устройством.
    :param command: Команда ('up', 'maxup', 'down').
    """
    angle = 0
    if command == 'up': angle = 90
    elif command == 'maxup': angle = 0
    elif command == 'down': angle = down
    ser.write(f'M280 P0 S{angle}\n'.encode())
    ser.read_until(b'ok\n')
        
def generate_gcode(trajectory: list):
    """
    Генерирует G-код на основе траектории движения.
    
    :param trajectory: Список координат траектории.
    :return: Список команд G-кода.
    """
    i = 2
    gcode_commands = [f'G1 X{(trajectory[0][0])} Y{(trajectory[0][1])} F{speed * 2}\n', 'down']
    while i < len(trajectory):
        while i < len(trajectory) and trajectory[i][0] != 1e9:
            if abs(trajectory[i][0]) != 1e9:
                gcode_commands += [f'G1 X{(trajectory[i][0])} Y{(trajectory[i][1])} F{speed}\n']
            i += 1
        gcode_commands += ['up']
        if i < len(trajectory) - 1 and abs(trajectory[i + 1][0]) != 1e9:
            gcode_commands += [f'G1 X{(trajectory[i + 1][0])} Y{(trajectory[i + 1][1])} F{speed * 2}\n']
        gcode_commands += ['down']
        i += 3
    return gcode_commands

def send_gcode(gcode_commands: list):
    """
    Отправляет G-код на устройство через последовательный порт.
    
    :param gcode_commands: Список команд G-кода.
    """
    vid = cv2.VideoCapture(0)
    ser = serial.Serial(port, baudrate=baudrate)
    sleep(2)
    ser.write(b'G90\n')
    calibrate(ser)
    prev_x, prev_y = 0, 0
    timestamp = 0
    i = 0
    try:
        flag_stop = False
        with HandLandmarker.create_from_options(options) as landmarker:
            while i < len(gcode_commands):
                timestamp += 1
                res, img = vid.read()
                if not res:
                    print('Fix the cam')
                    continue
                detection = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), timestamp)
                if detection.hand_landmarks:
                    if not flag_stop: 
                        print('Hand in the working space')
                        print('Stopped painting')
                    continue
                else: flag_stop = False
                gcode = gcode_commands[i]
                i += 1
                print(gcode)
                if (gcode[0] != 'G'):
                    control_servo(ser, gcode)
                    sleep(0.1)
                    continue
                if gcode == f"G1 X{prev_x} Y{prev_y} F{speed}": 
                    continue
                ser.write(gcode.encode())
                ser.read_until(b'ok\n')
                gcode = [float(gcode[gcode.index('X') + 1:gcode.index('Y') - 1]), float(gcode[gcode.index('Y') + 1:gcode.index('F') - 1])]
                d = calculate_distance(prev_x, prev_y, gcode[0], gcode[1]) / (speed / 60)
                sleep(d + 0.1)
                prev_x, prev_y = gcode
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
    control_servo(ser, 'maxup')
    ser.write('G1 X0 Y0 F16000\n'.encode())
    ser.close()

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

if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyACM0', 115200)
    sleep(2)
    ser.write(b'G90\n')
    print(ser.read_until(b'ok\n').decode())
    calibrate(ser)
    try:
        while True:
            command = input('Enter your command: ')
            if command[0] == 'X' or command[0] == 'Y':
                ser.write(('G1 ' + command + '\n').encode())
            elif command in ['up', 'down', 'maxup']:
                control_servo(ser, command)
            else:
                ser.write((command + '\n').encode())
            print(ser.read_until(b'ok\n').decode())
                                                                
    except KeyboardInterrupt:
        control_servo(ser, 'maxup')
        ser.write('G1 X0 Y0 F16000\n'.encode())
    ser.close()