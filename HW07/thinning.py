import cv2
import numpy as np
image = cv2.imread('lena.bmp', 0)
threshold, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
h, w = image.shape

# down sample
downsample = np.zeros((h//8, w//8), np.uint8) 
for i in range(0, h, 8):
    for j in range(0, w, 8):
        downsample[i//8][j//8] = image[i][j]
hh, ww = downsample.shape
cv2.imwrite("downsample.bmp", downsample)

def pixelValues(img, i, j):
    neighborhood = []
    for x in range(9):
        if x == 0:
            neighborhood.append(img[i][j])
        elif x == 1:
            if j+1 >= ww:
                neighborhood.append(0)
            else:
                neighborhood.append(img[i][j+1])
        elif x == 2:
            if i-1 < 0:
                neighborhood.append(0)
            else: 
                neighborhood.append(img[i-1][j])
        elif x == 3:
            if j-1 < 0:
                neighborhood.append(0)
            else:
                neighborhood.append(img[i][j-1])
        elif x == 4:
            if i+1 >= hh:
                neighborhood.append(0)
            else:
                neighborhood.append(img[i+1][j])
        elif x == 5:
            if i+1 >= hh or j+1 >= ww:
                neighborhood.append(0)
            else:
                neighborhood.append(img[i+1][j+1])
        elif x == 6:
            if i-1 < 0 or j+1 >= ww:
                neighborhood.append(0)
            else:
                neighborhood.append(img[i-1][j+1])
        elif x == 7:
            if i-1 < 0 or j-1 < 0:
                neighborhood.append(0)
            else:
                neighborhood.append(img[i-1][j-1])
        elif x == 8:
            if i+1 >= hh or j-1 < 0:
                neighborhood.append(0)
            else:
                neighborhood.append(img[i+1][j-1])
    return neighborhood

# Yokoi connectivity number
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

def Yokoi():
    for i in range(hh):
        for j in range(ww):
            if downsample[i][j] == 0:
                yokoi[i][j] = 0
                continue
            x = pixelValues(downsample, i, j) # find x0 ~ x8's pixel value
            a1 = h(x[0], x[1], x[6], x[2])
            a2 = h(x[0], x[2], x[7], x[3])
            a3 = h(x[0], x[3], x[8], x[4])
            a4 = h(x[0], x[4], x[5], x[1])
            yokoi[i][j] = f(a1, a2, a3, a4)

# Pair relationship operator
def H(a, m):
    if a == m:
        return 1
    return 0

def y(x, m):
    sum = 0
    for i in range(1, 5):
        sum += H(x[i], m)

    if sum < 1 or x[0] != m:
        return 'q'
    elif sum >= 1 and x[0] == m:
        return 'p'
    return ''

def pairOperator(): 
    for i in range(hh):
        for j in range(ww):
            if yokoi[i][j] == 0:
                continue
            x = pixelValues(yokoi, i, j) # find x0 ~ x8's pixel value
            pair[i][j] = y(x, 1)

# Connected Shrink Operator
def h_connected(b, c, d, e):
    if b == c and (d != b or e !=b):
        return 1
    return 0

def f_connected(a1, a2, a3, a4, x0):
    if sum([a1, a2, a3, a4]) == 1:
        return 0 # background
    return x0

# Thinning Operator
def thinning():
    Yokoi()
    pairOperator()
    for i in range(hh):
        for j in range(ww):
            if pair[i][j] != 'p':
                continue
            x = pixelValues(downsample, i, j) # find x0 ~ x8's pixel value
            a1 = h_connected(x[0], x[1], x[6], x[2])
            a2 = h_connected(x[0], x[2], x[7], x[3])
            a3 = h_connected(x[0], x[3], x[8], x[4])
            a4 = h_connected(x[0], x[4], x[5], x[1])
            downsample[i][j] = f_connected(a1, a2, a3, a4, x[0])

yokoi = np.zeros((hh, ww), np.uint8)
pair = [['' for _ in range(ww)] for _ in range(hh)]
# 7 iteration
for i in range(7):
    thinning()

cv2.imshow('Output', downsample)
cv2.waitKey(0)                             
cv2.destroyAllWindows()
cv2.imwrite("thinning.bmp", downsample)