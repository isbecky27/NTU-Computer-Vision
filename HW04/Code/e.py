# Hit-and-miss transform
import cv2
import numpy as np
img = cv2.imread('lena.bmp',0)
threshold, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
h, w = img.shape

J = np.array([[0, 0, 0],
              [1, 1, 0],
              [0, 1, 0]])

K = np.array([[0, 1, 1],
              [0, 0, 1],
              [0, 0, 0]])

def erosion(img, kernel):
    hh, ww = kernel.shape
    hh_half, ww_half = hh//2, ww//2
    output = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            flag = 0
            for ii in range(hh):
                for jj in range(ww):
                    if kernel[ii][jj] == 1:
                        if i+ii-hh_half < 0 or j+jj-ww_half < 0 or i+ii-hh_half >= h or j+jj-ww_half >= w:
                            continue 
                        if img[i+ii-hh_half][j+jj-ww_half] != 255:
                            flag = 1
                            break
            if flag == 0:
                output[i][j] = 255
    return output

img_J = erosion(img, J)
img_K = erosion(255-img, K)

output = np.zeros((h, w), np.uint8)
for i in range(h):
    for j in range(w):
        if img_J[i][j] == 255 and img_K[i][j] == 255:
            output[i][j] = 255

cv2.imshow('Hit-and-miss', output)
cv2.waitKey(0)                             
cv2.destroyAllWindows()

cv2.imwrite('e_hit-and-miss.png', output)