from get_trajectory import draw_img
from skimage.transform import resize
from imageio.v3 import imread
from time import sleep

robot = imread('images/robot.png')
robot = resize(robot, (512, robot.shape[1] * (512 / robot.shape[0])))[:, ::-1]

rro = imread('images/RRO.png')
rro = resize(rro, (512, rro.shape[1] * (512 / rro.shape[0])))

# A5
# draw_img(robot, k=512/180, deltax=390, deltay=290)
# sleep(5)
# draw_img(rro, k=512/140, deltax=406, deltay=20)

# A3
draw_img(robot, k=512/180, deltax=390, deltay=590)
sleep(5)
draw_img(rro, k=512/140, deltax=426, deltay=10)  
