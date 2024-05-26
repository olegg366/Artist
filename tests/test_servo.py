import serial
from time import sleep

ser = serial.Serial('/dev/ttyACM0', 115200)
sleep(2)
ser.write(b'G90\n')
try:
    while True:
        n = input()
        if n == 'up':
            ser.write(b'M42 P12 S255 T1\n')
            sleep(1000 / 1000000)
            ser.write(b'M42 P12 S0 T1\n')
        elif n == 'down':
            for i in range(180):
                ser.write(b'M42 P12 S255 T1\n')
                sleep(2000 / 1000000)
                ser.write(b'M42 P12 S0 T1\n')
        elif n[0] == 'X' or n[0] == 'Y':
            ser.write(('G1 ' + n + '\n').encode())
        else:
            ser.write((n + '\n').encode())
except KeyboardInterrupt:
    pass
ser.close()