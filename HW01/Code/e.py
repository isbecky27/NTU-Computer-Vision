import cv2
img = cv2.imread('lena.bmp')

h, w, _ = img.shape

output = cv2.resize(img, (int(h/2), int(w/2)))

cv2.imshow('Result', output)
cv2.waitKey(0)                             
cv2.destroyAllWindows()

cv2.imwrite('e.png', output)