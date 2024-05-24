import serial
from time import sleep

ser = serial.Serial('/dev/ttyACM0', 115200)
sleep(2)
ser.write(b'G90\n')
try:
    while True:
        n = input()
        if n == 'up':
            for i in range(180):
                ser.write(b'M42 P12 S255 T1\n')
                sleep(1900 / 1000000)
                ser.write(b'M42 P12 S0 T1\n')
        elif n == 'down':
            for i in range(180):
                ser.write(b'M42 P12 S255 T1\n')
                sleep(2150 / 1000000)
                ser.write(b'M42 P12 S0 T1\n')
        else:
            ser.write(('G1 ' + n + '\n').encode())
        # sleep((20000 - n)/1000000)
except KeyboardInterrupt:
    pass
ser.close()