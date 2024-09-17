from imageio import imread, imsave
from skimage.transform import resize

img = imread('images/point_up.png')

img = resize(img, (245, 222))
imsave('images/point_up.png', (img * 255).astype('uint8'))