import numpy as np
import math
import cv2
import collections


def LevelLineAngle(x, y):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    if x < 0 or x >= grad_x.shape[0] or y < 0 or y >= grad_x.shape[1]:
        return None
    else:
        angle = math.atan2(grad_y[x, y], grad_x[x, y])
        with open("LevelLineAngle3.txt", "a") as f:
            f.write(f"({x}, {y}): {angle}\n")
        return angle


def region_grow(x, y):
    StartX, StartY = x, y
    region = set([(StartX, StartY)])  
    regionAngle = LevelLineAngle(StartX, StartY)
    sx = math.cos(regionAngle)
    sy = math.sin(regionAngle)
    tolerance = math.radians(22)  
    
    n=0
    while True: 
        px, py = StartX+round(sx), StartY
        angle = LevelLineAngle(px, py)
        if angle is not None and \
            angle != 0 :
            #abs(angle-regionAngle) <= tolerance:
                region.add((px, py))
                print(f"({px}, {py}, {angle})")
                StartX=px
                StartY=py
                n+=1
                continue
        else:
            break
    
    print(n)          
    while True: 
        px, py = StartX, StartY+round(sy)
        angle = LevelLineAngle(px, py)
        if angle is not None and \
            (px,py) not in region and \
            angle != 0 :
            #abs(angle-regionAngle) <= tolerance:
                region.add((px, py))
                print(f"({px}, {py}, {angle})")
                StartX=px
                StartY=py
                n+=1
                continue
        else:
            break
    
    return region  # 返回region集合


img = cv2.imread("lsd_test_3.png")
h, w = img.shape[:2]

StartX, StartY = 19,18
region = region_grow(StartX, StartY)

img_new = np.zeros((h, w, 3), dtype=np.uint8)
for x, y in region:
    img_new[x, y]=(0,0,255)

print(region)
img = cv2.resize(img, (0, 0), fx=10, fy=10)
img_new = cv2.resize(img_new, (0, 0), fx=10, fy=10)
result = cv2.addWeighted(img_new, 0.6, img, 0.4, 0)

cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
