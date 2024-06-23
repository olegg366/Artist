from get_trajectory import draw_img
from skimage.transform import resize
from imageio.v3 import imread
from time import sleep
from serial_control import servo
from serial import Serial

robot = imread('images/robot.png')
robot = resize(robot, (512, robot.shape[1] * (512 / robot.shape[0])))

rro = imread('images/RRO.png')
rro = resize(rro, (512, rro.shape[1] * (512 / rro.shape[0])))

draw_img(robot, deltay=390)
sleep(5)
draw_img(rro, k=512/140, deltay=406, deltax=robot.shape[0] / 512 * 180 + 70)