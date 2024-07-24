import numpy as np
import math
import cv2

def LevelLineAngle(x,y):
    gray=cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY) # Convert image to grayscale
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)

    #if 0 <= x < grad_x.shape[0] and 0 <= y < grad_x.shape[1]:
    angle = math.atan2(grad_y[x, y], grad_x[x, y])
    return angle
    """else:
        raise ValueError("x 或 y 超出影像範圍")"""

# create a image
h,w=100,100
img_new = np.zeros((h, w, 3), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        if i==j:
            img_new[i, j] = (0, 0, 255)
        else:
            img_new[i, j] = (255, 255, 255)
        """elif i==h-j:
            img_new[i, j] = (255, 0, 0)"""
        with open('LevelLineAngle.txt', 'a') as f:
            f.write(f"({i}, {j}): {LevelLineAngle(i, j)}\n")
            
region={
    (0, 0): {"status": "unused"},
}
regionAngle=LevelLineAngle(0, 1)
sx=math.cos(regionAngle)
sy=math.sin(regionAngle)

cv2.imshow('image', img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()