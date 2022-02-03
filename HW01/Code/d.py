from PIL import Image
import cv2
import numpy as np

# Method 1
def rotate_image(img, angle):
  image_center = tuple(np.array(img.shape[1::-1])/2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

img = cv2.imread('lena.bmp')

output = rotate_image(img, -45)

cv2.imshow('Result', output)
cv2.waitKey(0)                             
cv2.destroyAllWindows()

cv2.imwrite('d.png', output)

# Method 2
img = Image.open("./lena.bmp")
output = img.rotate(-45)

output.show()
output.save('d.png')