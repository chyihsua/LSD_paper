import numpy as np
import math
import cv2

def LevelLineAngle(x, y):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)

    if x < 0 or x >= grad_x.shape[0] or y < 0 or y >= grad_x.shape[1]:
        return None
    else:
        angle = math.atan2(grad_y[x, y], grad_x[x, y])
        with open("LevelLineAngle2.txt", "a") as f:
            f.write(f"({x}, {y}): {angle}\n")
        return angle

# create a image
h, w = 100, 100
img = np.zeros((h, w, 3), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        if i == j:
            img[i, j] = (0, 0, 255)
        else:
            img[i, j] = (255, 255, 255)
        '''
        elif i==h-j:
            img[i, j] = (255, 0, 0)
        '''

region = {(w // 2, h // 2)}
regionAngle = LevelLineAngle(0, 0)
sx = math.cos(math.radians(regionAngle))
sy = math.sin(math.radians(regionAngle))
tolerance = math.radians(22.5)

max = max(h, w)
n = 0
while n <= max:
    for P in region.copy():  # for each pixel P in region
        # for p bar neighbor of P (8 neighbors)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                x = P[0] + dx
                y = P[1] + dy
                if (
                    (x >= 0 and x <= w)
                    and (y >= 0 and y <= h)  # ensure x, y is in the image
                    and
                    # check if (x, y) is not used by other region(X)
                    LevelLineAngle(x, y) != None
                    and
                    # if Diff(LevelLineAngle(x, y), regionAngle) < tolerance
                    abs(LevelLineAngle(x, y) - regionAngle) <= tolerance
                ):
                    # region[P] = {"status": "used"}
                    # region[(x, y)] = {"status": "unused"}
                    region.add((x, y))
                    sx += math.cos(math.radians(LevelLineAngle(x, y)))
                    sy += math.sin(math.radians(LevelLineAngle(x, y)))
                    regionAngle = math.degrees(math.atan2(sy, sx))
    n += 1

print(region)
img_new = np.zeros((h, w, 3), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        img_new[i, j] = (255, 255, 255)

# Draw the region
for k in region:
    img_new[k[0], k[1]] = (0, 255, 0)

img = cv2.resize(img, (0, 0), fx=10, fy=10)
img_new = cv2.resize(img_new, (0, 0), fx=10, fy=10)
result = cv2.addWeighted(img_new, 0.6, img, 0.4, 0)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()