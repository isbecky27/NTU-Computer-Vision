import cv2
img = cv2.imread('lena.bmp', 0)

h = img.shape[0]
w = img.shape[1]

output = img.copy()

for i in range(h):
    for j in range(w):
        if img[i][j] >= 128:
            output[i][j] = 255
        else:
            output[i][j] = 0

cv2.imshow('Result', output)
cv2.waitKey(0)                             
cv2.destroyAllWindows()

cv2.imwrite('f.png', output)