# Erosion
import cv2
import numpy as np
img = cv2.imread('lena.bmp',0)
threshold, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

h = img.shape[0]
w = img.shape[1]

kernel = np.array([[0, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0]])
hh, ww = kernel.shape

output = np.zeros((h, w), np.uint8) 
for i in range(h):
    for j in range(w):
        flag = 0
        for ii in range(hh):
            for jj in range(ww):
                if kernel[ii][jj] == 1:
                    if i+ii-2 < 0 or j+jj-2 < 0 or i+ii-2 >= h or j+jj-2 >= w:
                        continue 
                    if img[i+ii-2][j+jj-2] != 255:
                        flag = 1
                        break
        if flag == 0:
            output[i][j] = 255

cv2.imshow('Erosion', output)
cv2.waitKey(0)                             
cv2.destroyAllWindows()

cv2.imwrite('b_erosion.png', output)