import serial
from time import sleep
import os

os.system('echo Dragon2009 | sudo -S chmod 666 /dev/ttyACM0')

port = "/dev/ttyACM0"  
baudrate = 115200 

speed = 8000

def servo(ser, n):
    if n == 'up':
        ser.write('M280 P0 S90\n'.encode())
    else:
        ser.write('M280 P0 S135\n'.encode())
    ser.read_until(b'ok\n')
        
def get_gcode(t: list):
    i = 2
    k = 1
    ans = [f'G1 X{t[0][0] / k} Y{t[0][1] / k} F{speed * 2}\n', 'down']
    while i < len(t):
        while i < len(t) and t[i][0] != 1e9:
            if abs(t[i][0]) != 1e9:
                ans += [f'G1 X{t[i][0] / k} Y{t[i][1] / k} F{speed}\n']
            i += 1
        ans += ['up']
        if i < len(t) - 1 and abs(t[i + 1][0]) != 1e9:
            ans += [f'G1 X{t[i + 1][0] / k} Y{t[i + 1][1] / k} F{speed * 2}\n']
        ans += ['down']
        i += 3
    return ans

def send_gcode(gcodes: list):
    ser = serial.Serial(port, baudrate=baudrate)
    sleep(2)
    ser.write(b'G90\n')
    prevx, prevy = 0, 0
    try:
        for gcode in gcodes:
            print(gcode)
            if (gcode[0] != 'G'):
                servo(ser, gcode)
                sleep(0.2)
                continue
            if gcode == f"G1 X{prevx} Y{prevy} F{speed}": 
                continue
            ser.write(gcode.encode())
            ser.read_until(b'ok\n')
            gcode = [float(gcode[gcode.index('X') + 1:gcode.index('Y') - 1]), float(gcode[gcode.index('Y') + 1:gcode.index('F') - 1])]
            d = dist(prevx, prevy, gcode[0], gcode[1]) / (speed / 60)
            sleep(d + 0.2)
            prevx, prevy = gcode[0], gcode[1]
    except KeyboardInterrupt:
        pass
    servo(ser, 'up')
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
            elif n in ['up', 'down']:
                servo(ser, n)
                
    except KeyboardInterrupt:
        servo(ser, 'up')
        ser.write('G1 X0 Y0 F16000\n'.encode())
    ser.close()