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
    region = set()
    regionAngle = LevelLineAngle(StartX, StartY)
    sx = math.cos(regionAngle)
    sy = math.sin(regionAngle)
    tolerance = 22.5
    quene = collections.deque()
    quene.append((StartX, StartY))

    # for p bar neighbor of P (8 neighbors)
    while quene:
        #print(quene)
        i,j = quene.popleft()
        region.add((i,j))
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                x = StartX + dx
                y = StartY + dy
                if (
                    (x >= 0 and x <= w)
                    and (y >= 0 and y <= h)  # ensure x, y is in the image
                    and LevelLineAngle(x, y) != None
                    and
                    # check if (x, y) is not used by other region
                    status[x, y] == 0
                    and
                    # if Diff(LevelLineAngle(x, y), regionAngle) < tolerance
                    abs(LevelLineAngle(x, y) - regionAngle) <= tolerance
                ):
                    quene.append((x, y))
                    sx += math.cos(LevelLineAngle(x, y))
                    sy += math.sin(LevelLineAngle(x, y))
                    regionAngle = math.atan2(sy, sx)
                    StartX, StartY = x, y
    for i in range(len(region)):
        status[region[i][0], region[i][1]] = 1


img = cv2.imread("lsd_test_4.png")
h, w = img.shape[:2]

status = np.zeros((h, w), dtype=int)
for i in range(h):
    for j in range(w):
        if status[i, j] == 0:
            print(i, j)
            region_grow(i, j)
            print(region_grow(i, j))
        else:
            break

img_new = np.zeros((h, w, 3), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        if status[i, j] == 1:
            img_new[i, j] = (0, 255, 0)
        else:
            img_new[i, j] = (255, 255, 255)

print(status)
img = cv2.resize(img, (0, 0), fx=10, fy=10)
img_new = cv2.resize(img_new, (0, 0), fx=10, fy=10)
result = cv2.addWeighted(img_new, 0.6, img, 0.4, 0)

cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
