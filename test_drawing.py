import pickle
from main import draw_img, get_gcode, send_gcode
from PIL import Image
import numpy as np
import cv2

img = Image.open('now.png')

img = Image.fromarray(np.array(img)[:, :, :3])

draw_img(img)