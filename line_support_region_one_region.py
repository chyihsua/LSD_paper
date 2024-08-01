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
        with open("LevelLineAngle2.txt", "a") as f:
            f.write(f"({x}, {y}): {angle}\n")
        return angle

def region(x,y):
    StartX,StartY = x,y
    region = {(StartX,StartY)}
    regionAngle = LevelLineAngle(StartX,StartY)
    sx = math.cos(math.radians(regionAngle))
    sy = math.sin(math.radians(regionAngle))
    tolerance = math.radians(22.5)
    quene = collections.deque()
    quene.append((StartX,StartY))

    while quene:
        i,j = quene.popleft()
        region.add((i,j))
        status[i,j] = 1
        
            # for p bar neighbor of P (8 neighbors)
        for dx in range(-1, 2):
            for dy in range(-1,2):
                    x = i + dx
                    y = j + dy
                    if (
                        (x >= 0 and x <= w)
                        and (y >= 0 and y <= h)  # ensure x, y is in the image
                        and
                        LevelLineAngle(x, y) != None and
                        #check if (x, y) is not used by other region
                        status[x, y] == 0 and
                        (x,y) not in region and
                        # if Diff(LevelLineAngle(x, y), regionAngle) < tolerance
                        abs(LevelLineAngle(x, y) - regionAngle) <= tolerance
                    ):  
                        region.add((x, y))
                        status[x, y] = 1
                        quene.append((x, y))
                        sx += math.cos(math.radians(LevelLineAngle(x, y)))
                        sy += math.sin(math.radians(LevelLineAngle(x, y)))
                        regionAngle = math.degrees(math.atan2(sy, sx))
                    #elif (x,y) in region:
                    #    quene.remove((x,y))


# create a image
'''
h, w = 16,16
img = np.zeros((h, w, 3), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        if i == j:
            img[i, j] = (0, 0, 255)
        
        else:
            img[i, j] = (255, 255, 255)
        
        elif i==h-j:
            img[i, j] = (255, 0, 0)
'''

img = cv2.imread("/Users/chyihsua/Desktop/LSD_paper/lsd_test_3.png")
h, w = img.shape[:2]

# 建立一個 h x w 的矩陣，初始值都是 0
status = np.zeros((h, w), dtype=int)
for i in range(h):
    for j in range(w):
        if status[i, j] == 0:
            region(i,j)
            #print(region)
        else:
            break

img_new = np.zeros((h, w, 3), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        if status[i, j] == 1:
            img_new[i, j] = (128, 255, 0)
        else:
            img_new[i, j] = (255, 255, 255)
print(status)
'''
for i in range(h):
    for j in range(w):
        img_new[i, j] = (255, 255, 255)
'''
# 顯示矩陣

'''
# Draw the region
for k in region:
    #cv2.rectangle(img, (i, j), (i , j ), (0, 255, 0), -1)
    img_new[k[0],k[1]] = (0, 255, 0)
'''
#img = cv2.resize(img, (0, 0), fx=10, fy=10)
#img_new = cv2.resize(img_new, (0, 0), fx=10, fy=10)
#result = cv2.addWeighted(img_new, 0.6, img, 0.4, 0)

cv2.imshow("result", img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()

