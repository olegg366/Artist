from get_trajectory import draw_img
from skimage.transform import resize
from imageio.v3 import imread
from time import sleep

robot = imread('images/robot.png')
robot = resize(robot, (512, robot.shape[1] * (512 / robot.shape[0])))[:, ::-1]

olymp = imread('images/robofinist.png')
olymp = resize(olymp, (512, olymp.shape[1] * (512 / olymp.shape[0])))

# A5
def draw_a5():
    # draw_img(robot, crop=True, show=True)
    # sleep(5)
    draw_img(olymp, crop=True, show=True)
    
if __name__ == '__main__':
    draw_a5()
