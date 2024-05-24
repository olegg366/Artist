import serial

ser = serial.Serial('COM1', 112500)

try:
    while True:
        inp = input()
        ser.write(inp.encode())
except KeyboardInterrupt:
    pass
ser.close()
    
