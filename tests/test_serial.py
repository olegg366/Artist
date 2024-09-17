import serial
from time import sleep
import os

os.system('echo Dragon2009 | sudo -S chmod 666 /dev/ttyACM0')

port = "/dev/ttyACM0"  
baudrate = 115200 

speed = 8000

ser = serial.Serial(port, baudrate)
sleep(2)
try:
    while True:
        s = input() + '\n'
        ser.write(s.encode())
        res = ser.read_until(b'ok\n')
        print(res.decode())
except KeyboardInterrupt:
    pass
ser.close()