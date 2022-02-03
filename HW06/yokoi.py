import cv2
import numpy as np
img = cv2.imread('lena.bmp', 0)
threshold, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
h, w = img.shape

# down sample
downsample = np.zeros((h//8, w//8), np.uint8) 
for i in range(0, h, 8):
    for j in range(0, w, 8):
        downsample[i//8][j//8] = img[i][j]
hh, ww = downsample.shape

# cv2.imshow('Downsampled', downsample)
# cv2.waitKey(0)                             
# cv2.destroyAllWindows()
# cv2.imwrite("downsample.bmp", downsample)

# Yokoi connectivity number
def pixelValues(i, j):
    neighborhood = []
    for x in range(9):
        if x == 0:
            neighborhood.append(downsample[i][j])
        elif x == 1:
            if j+1 >= ww:
                neighborhood.append(0)
            else:
                neighborhood.append(downsample[i][j+1])
        elif x == 2:
            if i-1 < 0:
                neighborhood.append(0)
            else: 
                neighborhood.append(downsample[i-1][j])
        elif x == 3:
            if j-1 < 0:
                neighborhood.append(0)
            else:
                neighborhood.append(downsample[i][j-1])
        elif x == 4:
            if i+1 >= hh:
                neighborhood.append(0)
            else:
                neighborhood.append(downsample[i+1][j])
        elif x == 5:
            if i+1 >= hh or j+1 >= ww:
                neighborhood.append(0)
            else:
                neighborhood.append(downsample[i+1][j+1])
        elif x == 6:
            if i-1 < 0 or j+1 >= ww:
                neighborhood.append(0)
            else:
                neighborhood.append(downsample[i-1][j+1])
        elif x == 7:
            if i-1 < 0 or j-1 < 0:
                neighborhood.append(0)
            else:
                neighborhood.append(downsample[i-1][j-1])
        elif x == 8:
            if i+1 >= hh or j-1 < 0:
                neighborhood.append(0)
            else:
                neighborhood.append(downsample[i+1][j-1])
    return neighborhood


def h(b, c, d, e):
    if b == c and (d != b or e != b):
        return 'q'
    elif b == c and (d == b and e == b):
        return 'r'
    elif b != c:
        return 's'

def f(a1, a2, a3, a4):
    if a1 == a2 == a3 == a4 == 'r':
        return 5
    else:
        return [a1, a2, a3, a4].count('q')

output = np.zeros((hh, ww), np.uint8)
for i in range(hh):
    for j in range(ww):
        if downsample[i][j] == 0:
            output[i][j] = 0
            continue
        x = pixelValues(i, j) # find x0 ~ x8's pixel value
        a1 = h(x[0], x[1], x[6], x[2])
        a2 = h(x[0], x[2], x[7], x[3])
        a3 = h(x[0], x[3], x[8], x[4])
        a4 = h(x[0], x[4], x[5], x[1])
        output[i][j] = f(a1, a2, a3, a4)

for i in range(hh):
    for j in range(ww):
        if output[i][j] == 0:
            print("  ", end='')
        else:
            print("{:2d}".format(output[i][j]), end='')
    print("")