import cv2
img = cv2.imread('lena.bmp')

h = img.shape[0]
w = img.shape[1]

output = img.copy()

for i in range(h):
    for j in range(w):
        output[h-1-i][j] = img[i][j]

cv2.imshow('Result', output)
cv2.waitKey(0)                             
cv2.destroyAllWindows()

cv2.imwrite('a.png', output)