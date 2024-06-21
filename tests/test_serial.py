import serial

import os

os.system('echo Dragon2009 | sudo -S chmod 666 /dev/ttyACM0')

port = "/dev/ttyACM0"  
baudrate = 115200 

speed = 8000

ser = serial.Serial(port, baudrate)

def recvFromArduino():
  global startMarker, endMarker
  
  ck = ""
  x = "z" # any value that is not an end- or startMarker
  byteCount = -1 # to allow for the fact that the last increment will be one too many
  
  while  ord(x) != startMarker: 
    x = ser.read()
  
  # save data until the end marker is found
  while ord(x) != endMarker:
    if ord(x) != startMarker:
      ck = ck + x 
      byteCount += 1
    x = ser.read()
  
  return(ck)