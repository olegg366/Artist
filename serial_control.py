import serial
from time import sleep
import mediapipe as mp
import numpy  as np
from math import sin, cos
import os
import cv2

os.system('echo Dragon2009 | sudo -S chmod 666 /dev/ttyACM0')

port = "/dev/ttyACM0"  
baudrate = 115200 

speed = 8000

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='mlmodels/hand_landmarker.task', delegate=BaseOptions.Delegate.CPU),
        running_mode=VisionRunningMode.VIDEO)

def servo(ser: serial.Serial, n):
    angle = 0
    if n == 'up': angle = 90
    else: angle = 135
    ser.write(f'M280 P0 S{angle}\n'.encode())
    ser.read_until(b'ok\n')
    ser.write('M280 P0\n'.encode())
    pos = ser.read_all().decode()
    while pos != angle:
        ser.write('M280 P0\n'.encode())
        ser.read_until(b'ok\n')
        pos = ser.read_all().decode()
        
def get_gcode(t: list, k=1, deltax=0, deltay=0):
    i = 2
    ans = [f'G1 X{(t[0][0] / k) + deltax} Y{(t[0][1] / k) + deltay} F{speed * 2}\n', 'down']
    while i < len(t):
        while i < len(t) and t[i][0] != 1e9:
            if abs(t[i][0]) != 1e9:
                ans += [f'G1 X{(t[i][0] / k) + deltax} Y{(t[i][1] / k) + deltay} F{speed}\n']
            i += 1
        ans += ['up']
        if i < len(t) - 1 and abs(t[i + 1][0]) != 1e9:
            ans += [f'G1 X{(t[i + 1][0] / k) + deltax} Y{(t[i + 1][1] / k) + deltay} F{speed * 2}\n']
        ans += ['down']
        i += 3
    return ans

def shift_matrix(sx: float, sy: float, angle: float, matrix: np.ndarray):
    rm = np.array([[cos(angle), sin(angle)],
                   [-sin(angle), cos(angle)]])
    rotated = np.matmul(matrix, rm)
    shifted = rotated + [sx, sy]
    return shifted

def send_gcode(gcodes: list):
    vid = cv2.VideoCapture(0)
    ser = serial.Serial(port, baudrate=baudrate)
    sleep(2)
    ser.write(b'G90\n')
    prevx, prevy = 0, 0
    try:
        flag_stop = False
        with HandLandmarker.create_from_options(options) as landmarker:
            for gcode in gcodes:
                res, img = vid.read()
                if not res:
                    print('Fix the cam')
                    continue
                detection = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), timestamp)
                if detection.hand_landmarks:
                    print('Hand in the working space')
                    if not flag_stop: 
                        flag_stop = True
                        ser.write(b'M0\n')
                        ser.read_until(b'ok\n')
                        print('Stopped painting')
                    continue
                print(gcode)
                if (gcode[0] != 'G'):
                    servo(ser, gcode)
                    # sleep(0.1)
                    continue
                if gcode == f"G1 X{prevx} Y{prevy} F{speed}": 
                    continue
                ser.write(gcode.encode())
                ser.read_until(b'ok\n')
                gcode = [float(gcode[gcode.index('X') + 1:gcode.index('Y') - 1]), float(gcode[gcode.index('Y') + 1:gcode.index('F') - 1])]
                d = dist(prevx, prevy, gcode[0], gcode[1]) / (speed / 60)
                # sleep(d + 0.1)
                prevx, prevy = gcode
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
    servo(ser, 'maxup')
    ser.write('G1 X0 Y0 F16000\n'.encode())
    ser.close()

def dist(ax, ay, bx, by):
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyACM0', 115200)
    sleep(2)
    ser.write(b'G90\n')
    try:
        while True:
            n = input()
            if n[0] == 'X' or n[0] == 'Y':
                ser.write(('G1 ' + n + '\n').encode())
            elif n in ['up', 'down', 'maxup']:
                servo(ser, n)
            else:
                ser.write((n + '\n').encode())
                
    except KeyboardInterrupt:
        servo(ser, 'maxup')
        ser.write('G1 X0 Y0 F16000\n'.encode())
    ser.close()