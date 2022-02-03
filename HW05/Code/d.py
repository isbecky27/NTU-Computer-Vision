# Closing
import cv2
import numpy as np
img = cv2.imread('lena.bmp', 0)

h = img.shape[0]
w = img.shape[1]

kernel = np.array([[0, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0]])
hh, ww = kernel.shape
hh_half, ww_half = hh//2, ww//2

def erosion(img):
    output = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            value = []
            for ii in range(hh):
                for jj in range(ww):
                    if kernel[ii][jj] == 1:
                        if i+ii-hh_half < 0 or j+jj-ww_half < 0 or i+ii-hh_half >= h or j+jj-ww_half >= w:
                            continue 
                        value.append(img[i+ii-hh_half][j+jj-ww_half])
            output[i][j] = min(value)
    return output

def dilation(img):
    output = np.zeros((h, w), np.uint8) 
    for i in range(h):
        for j in range(w):
            value = []
            for ii in range(hh):
                for jj in range(ww):
                    if kernel[ii][jj] == 1:
                        if i+ii-hh_half < 0 or j+jj-ww_half < 0 or i+ii-hh_half >= h or j+jj-ww_half >= w:
                            continue 
                        value.append(img[i+ii-hh_half][j+jj-ww_half])
            output[i][j] = max(value)
    return output

dilation_img = dilation(img)
closing_img = erosion(dilation_img)

cv2.imshow('Closing', closing_img)
cv2.waitKey(0)                             
cv2.destroyAllWindows()

cv2.imwrite('d_closing.png', closing_img)