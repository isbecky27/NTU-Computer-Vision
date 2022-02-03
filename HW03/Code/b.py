import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lena.bmp', 0)
h, w = img.shape

histogramX = [i for i in range(256)]
histogramY = [0 for _ in range(256)]

for i in range(h):
    for j in range(w):
        histogramY[int(img[i][j]/3)] += 1
        img[i][j] = int(img[i][j]/3)
# print(histogramY)

plt.bar(histogramX, histogramY, width=1)
plt.savefig("b_histogram.png") 
plt.show()

cv2.imshow('result', img)
cv2.waitKey(0)                             
cv2.destroyAllWindows()
cv2.imwrite('b_image.png', img)