import os
import pyautogui as pg, mouse as ms

cords = [[i, j] for i in range(500, 1000) for j in range(500, 1000)]

pg.FAILSAFE = False

def drag(x, y):
    os.system('xdotool mousedown 1')
    ms.move(x, y)
    os.system('xdotool mouseup 1')

for x, y in cords:
    pg.dragTo(x, y, 0.0, _pause=False)