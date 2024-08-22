import cv2
import numpy as np
import math

img=cv2.imread("lsd_test_3.png")
h, w = img.shape[:2]
region = set([(19, 18)])

for x in range(h):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for y in range(w):
        if not np.all(gray_img[x, y] >= 200):
            region.add((x, y))  #找到有色區域

#排序
region = sorted(region, key=lambda point: (point[0], point[1]))
#確認找到的座標
img_new = np.zeros((h, w, 3), dtype=np.uint8)
#找邊緣座標
max_x = float('-inf')
max_y = float('-inf')
min_x = float('inf')
min_y = float('inf')
    
for x, y in region:
    #img_new[x, y] = (255,255,255)
    if x > max_x:
        max_x = x
    if y > max_y:
        max_y = y
    if x < min_x:
        min_x = x
    if y < min_y:
        min_y = y

orientation = (max_x - min_x) / (max_y - min_y)
print("orientation",orientation)
angle = np.arctan(orientation) * 180 / np.pi
print("angle",angle)    #以x軸為基準的角度

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
    
#計算每一個region
for x in range(h):
    for y in range(w):
        if abs(LevelLineAngle(x, y) - orientation) > orientation:   
        #不確定t值設定標準對不對，不知道要設什麼
            img_new[x, y] = (0, 255, 0)

#畫出邊緣
#避免重疊   (條件有等於嗎？)
if min_x-1>0 and min_x-1<h:
    min_x-=1
if max_x+1>0 and max_x+1<h:
    max_x+=1
if min_y-1>0 and min_y-1<w:
    min_y-=1
if max_y+1>0 and max_y+1<w:
    max_y+=1
img_new[min_x,min_y:max_y] = (0,0,255)
img_new[max_x,min_y:max_y] = (0,0,255)
img_new[min_x:max_x,min_y] = (0,0,255)
img_new[min_x:max_x,max_y] = (0,0,255)

for x, y in region:
    img_new[x, y] = (255,255,255)

img_new = cv2.resize(img_new, (w*15, h*15))
cv2.imshow("img", img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()