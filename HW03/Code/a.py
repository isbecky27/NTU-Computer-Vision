import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lena.bmp', 0)
cv2.imwrite('a_image.png', img)

h, w = img.shape

histogramX = [i for i in range(256)]
histogramY = [0 for _ in range(256)]

for i in range(h):
    for j in range(w):
        histogramY[img[i][j]] += 1
# print(histogramY)

plt.bar(histogramX, histogramY, width=1)
plt.savefig("a_histogram.png") 
plt.show()


