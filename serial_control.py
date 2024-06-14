import serial
from time import sleep
import os

os.system('echo Dragon2009 | sudo -S chmod 666 /dev/ttyACM0')

DOWN = 2000
UP = 1000

port = "/dev/ttyACM0"  
baudrate = 115200 

speed = 5000

def servo(ser, n):
    # return None
    cnt = 10
    if n == UP:
        cnt = 1
    for i in range(cnt):
        ser.write(b'M42 P12 S255 T1\n')
        sleep(n / 1000000)
        ser.write(b'M42 P12 S0 T1\n')
        
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

def dist(ax, ay, bx, by):
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyACM0', 115200)
    sleep(2)
    ser.write(b'G90\n')
    try:
        while True:
            n = input()
            if n == 'up':
                servo(ser, UP)
            elif n == 'down':
                servo(ser, DOWN)
            elif n[0] == 'X' or n[0] == 'Y':
                ser.write(('G1 ' + n + '\n').encode())
            else:
                ser.write((n + '\n').encode())
    except KeyboardInterrupt:
        pass
    ser.close()