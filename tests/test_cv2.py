import cv2

vid = cv2.VideoCapture(0)

while True:
    res, img = vid.read()
    if not res: 
        print(0)
        continue
    print(img.shape)
    cv2.imshow('img', img)
    cv2.waitKey(1)