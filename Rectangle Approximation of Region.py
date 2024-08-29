import cv2
import numpy as np
import math

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

def region_grow():
    max_x = float('-inf')
    max_y = float('-inf')
    min_x = float('inf')
    min_y = float('inf')
    p1=(min_x,min_y)
    p2=(min_x,max_y)
    p3=(max_x,min_y)
    p4=(max_x,max_y)

    for x, y in region:
        if x <= p1[0] and y <= p1[1]:
            p1 = (x, y)
        elif x <= p2[0] and y >= p2[1]:
            p2 = (x, y)
        elif x >= p4[0] and y >= p4[1]:
            p4 = (x, y)
        elif x >= p3[0] and y <= p3[1]:
            p3 = (x, y)

    c1 = int((p1[0] + p3[0]) / 2), int((p1[1] + p3[1]) / 2)
    c2 = int((p2[0] + p4[0]) / 2), int((p2[1] + p4[1]) / 2)
    orientation = (c2[0] - c1[0]) / (c2[1] - c1[1])
    width=math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    length=math.sqrt((p1[0]-p3[0])**2+(p1[1]-p3[1])**2)
    angle = np.arctan(orientation) * 180 / np.pi

    for x in range(h):
        for y in range(w):
            if abs(LevelLineAngle(x, y) - orientation) <= orientation:   
            #不確定t值設定標準對不對，不知道要設什麼
                img_new[x, y] = (0, 255, 0)
            else:
                img_new[x, y] = (0, 0, 0)

    print(p1,p2,p3,p4,c1,c2)
    img_new[p1[0], p1[1]] = (255, 128, 255)
    img_new[p2[0], p2[1]] = (255, 128, 255)
    img_new[p3[0], p3[1]] = (255, 128, 255)
    img_new[p4[0], p4[1]] = (255, 128, 255)
    img_new[c1[0], c1[1]] = (255, 0, 255)
    img_new[c2[0], c2[1]] = (255, 0, 255)

img=cv2.imread("lsd_test_8.png")
h, w = img.shape[:2]
region = set([])
img_new = np.zeros((h, w, 3), dtype=np.uint8)

for x in range(h):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for y in range(w):
        if not np.all(gray_img[x, y] >= 200):
            region.add((x, y))  #找到有色區域

#排序
region = sorted(region, key=lambda point: (point[0], point[1]))

for i in range(len(region)):
    img_new[region[i][0], region[i][1]] = (255, 255, 255)

region_grow()

img_new=cv2.addWeighted(img, 0.5, img_new, 0.5, 0, img_new)
cv2.imshow("img_new", img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()