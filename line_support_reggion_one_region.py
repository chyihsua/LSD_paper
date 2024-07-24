import numpy as np
import math
import cv2

def LevelLineAngle(x, y):
    gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)

    if x < 0 or x >= grad_x.shape[0] or y < 0 or y >= grad_x.shape[1]:
        return None
    else:
        angle = math.atan2(grad_y[x, y], grad_x[x, y])
        with open('LevelLineAngle2.txt', 'a') as f:
                f.write(f"({x}, {y}): {angle}\n")
        return angle

# create a image
'''
h, w = 100, 100
img_new = np.zeros((h, w, 3), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        if i == j:
            img_new[i, j] = (0, 0, 255)
        else:
            img_new[i, j] = (255, 255, 255)
        elif i==h-j:
            img_new[i, j] = (255, 0, 0)
'''

img_new = cv2.imread("/Users/chyihsua/Downloads/LSD/lsd_test_3.png")
h,w=img_new.shape[:2]
print(h,w)

region = {
    (0, 0): {"status": "unused"},
}
regionAngle = LevelLineAngle(0,0)
sx = math.cos(math.radians(regionAngle))
sy = math.sin(math.radians(regionAngle))
tolerance = math.radians(22.5)

for P in region.copy():    # for each pixel P in region
    # for p bar neighbor of P (8 neighbors)
    for dx in range(0,w):
        for dy in range(0,h):
            x = P[0] + dx
            y = P[1] + dy
            if ((x >= 0 and x <= w) and #ensure x, y is in the image
                (y >= 0 and y <= h) and
                #check if (x, y) is not used
                ((x, y) not in region or region[(x, y)]["status"] == "unused") and  
                LevelLineAngle(x, y)!=None and
                #if Diff(LevelLineAngle(x, y), regionAngle) < tolerance
                abs(LevelLineAngle(x, y) - regionAngle) <= tolerance
            ):
                region[P] = {"status": "used"}
                region[(x, y)] = {"status": "unused"}
                sx += math.cos(math.radians(LevelLineAngle(x, y)))
                sy += math.sin(math.radians(LevelLineAngle(x, y)))
                regionAngle = math.degrees(math.atan2(sy, sx))
                #print(x,y,LevelLineAngle(x, y))
                print(x,y)

'''
for i in region.keys():
    if region[i]["status"] == "unused":
        print("unused region",i)
    else:
        print("used region",i)
'''

img = np.zeros((h, w, 3), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        img[i, j] = (255, 255, 255)

for k in region:
    cv2.circle(img, k, 1, (0, 255, 0), 0)

img_new = cv2.resize(img_new, (0, 0), fx=10, fy=10)
img = cv2.resize(img, (0, 0), fx=10, fy=10)
cv2.imshow("Original", img_new)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
