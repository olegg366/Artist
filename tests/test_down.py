import serial
from time import sleep
import os

os.system('echo Dragon2009 | sudo -S chmod 666 /dev/ttyACM0')
ser = serial.Serial("/dev/ttyACM0", 115200)
sleep(2)
angle = 0
while True:
    ser.write(f'M280 P0 S{angle}\n'.encode())
    ser.read_until(b'ok\n')
    angle += 5
    ser.write(b'M119\n')
    status = ser.read_until(b'ok\n').decode()
    state = status[status.rfind('x_min: ') + 7:status.find('\n', status.rfind('x_min: '))]
    print(state)
    if state == 'TRIGGERED' or angle == 180:
        break
sleep(5)
ser.write(b'M280 P0 S0\n')
ser.close()