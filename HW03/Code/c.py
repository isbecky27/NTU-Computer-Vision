import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('b_image.png', 0)
h, w = img.shape
total = h * w

histogramX = [i for i in range(256)]
histogramY = [0 for _ in range(256)]

for i in range(h):
    for j in range(w):
        histogramY[int(img[i][j])] += 1

accumulate = 0
for i in range(256):
    accumulate += histogramY[i]
    histogramY[i] = int(np.round(accumulate / total * 255))

output = img.copy()
histogramEqu = [0 for _ in range(256)]
for i in range(h):
    for j in range(w):
        output[i][j] = histogramY[int(img[i][j])]
        histogramEqu[output[i][j]] += 1

plt.bar(histogramX, histogramEqu, width=1)
plt.savefig("c_histogram.png") 
plt.show()

cv2.imshow('result', output)
cv2.waitKey(0)                             
cv2.destroyAllWindows()
cv2.imwrite('c_image.png', output)