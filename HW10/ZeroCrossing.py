import cv2
import math
import numpy as np
image = cv2.imread('lena.bmp', 0)
img_padding = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)
h, w = image.shape

def convolution(img, i, j, mask):
    rows, cols = mask.shape
    value = 0
    for a in range(rows):
        for b in range(cols):
            value += img[i+a][j+b] * mask[a][b]
    return value

def edgeDetection(img, threshold, kernel):
    output = np.zeros((h, w), np.int8)
    padding = kernel.shape[0] // 2
    img_padding = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REFLECT)
    for i in range(h):
        for j in range(w):
            gradient = convolution(img_padding, i, j, kernel)
            if gradient >= threshold:
                output[i][j] = 1
            elif gradient <= threshold * (-1): 
                output[i][j] = -1
            else:
                output[i][j] = 0
    output = zeroCrossing(output, padding)
    return output

def checkNeighbors(img, i, j):
    for a in range(-1, 2):
        for b in range(-1, 2):
            if img[i+a][j+b] == -1:
                return 1
    return 0

def zeroCrossing(img, padding):
    output = np.zeros((h, w), np.uint8)
    img_padding = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REFLECT)
    hh, ww = img_padding.shape
    for i in range(padding, hh-padding):
        for j in range(padding, ww-padding):      
            if img_padding[i][j] == 1:
                if checkNeighbors(img_padding, i, j) == 1: 
                    output[i-padding][j-padding] = 0
                    continue
            output[i-padding][j-padding] = 255
    return output

# Laplace Mask 1
Laplace_1 = np.array([[0, 1, 0],
                      [1,-4, 1],
                      [0, 1, 0]])

laplace1_img = edgeDetection(image, 15, Laplace_1)
cv2.imshow("Laplace Mask 1", laplace1_img)
cv2.waitKey(0)                             
cv2.destroyAllWindows()
cv2.imwrite("Laplace1_15.png", laplace1_img)

# Laplace Mask 2
Laplace_2 = np.array([[1, 1, 1],
                      [1,-8, 1],
                      [1, 1, 1]])
Laplace_2 = np.divide(Laplace_2, 3) # 1/3 * mask

laplace2_img = edgeDetection(image, 15, Laplace_2)
cv2.imshow("Laplace Mask 2", laplace2_img)
cv2.waitKey(0)                             
cv2.destroyAllWindows()
cv2.imwrite("Laplace2_15.png", laplace2_img)

# Minimum variance Laplacian
Laplace_min = np.array([[ 2, -1, 2],
                        [-1, -4,-1],
                        [ 2, -1, 2]])
Laplace_min = np.divide(Laplace_min, 3) # 1/3 * mask

laplaceMin_img = edgeDetection(image, 20, Laplace_min)
cv2.imshow("Minimum variance Laplacian", laplaceMin_img)
cv2.waitKey(0)                             
cv2.destroyAllWindows()
cv2.imwrite("LaplaceMin_20.png", laplaceMin_img)

# Laplace of Gaussian: 3000
LoG = np.array([[ 0, 0,  0, -1, -1, -2, -1, -1,  0, 0, 0],
                [ 0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
                [ 0,-2, -7,-15,-22,-23,-22,-15, -7,-2, 0],
                [-1,-4,-15,-24,-14, -1,-14,-24,-15,-4,-1],
                [-1,-8,-22,-14, 52,103, 52,-14,-22,-8,-1],
                [-2,-9,-23, -1,103,178,103, -1,-23,-9,-2],
                [-1,-8,-22,-14, 52,103, 52,-14,-22,-8,-1],
                [-1,-4,-15,-24,-14, -1,-14,-24,-15,-4,-1],
                [ 0,-2, -7,-15,-22,-23,-22,-15, -7,-2, 0],
                [ 0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
                [ 0, 0,  0, -1, -1, -2, -1, -1,  0, 0, 0]])

LoG_img = edgeDetection(image, 3000, LoG)
cv2.imshow("Laplace of Gaussian", LoG_img)
cv2.waitKey(0)                             
cv2.destroyAllWindows()
cv2.imwrite("LoG_3000.png", LoG_img)

# Difference of Gaussian
DoG = np.array([[-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
                [-3, -5, -8,-11,-13,-13,-13,-11, -8, -5, -3],
                [-4, -8,-12,-16,-17,-17,-17,-16,-12, -8, -4],
                [-6,-11,-16,-16,  0, 15,  0,-16,-16,-11, -6],
                [-7,-13,-17,  0, 85,160, 85,  0,-17,-13, -7],
                [-8,-13,-17, 15,160,283,160, 15,-17,-13, -8],
                [-7,-13,-17,  0, 85,160, 85,  0,-17,-13, -7],
                [-6,-11,-16,-16,  0, 15,  0,-16,-16,-11, -6],
                [-4, -8,-12,-16,-17,-17,-17,-16,-12, -8, -4],
                [-3, -5, -8,-11,-13,-13,-13,-11, -8, -5, -3],
                [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]])

DoG_img = edgeDetection(image, 1, DoG)
cv2.imshow("Difference of Gaussian", DoG_img)
cv2.waitKey(0)                             
cv2.destroyAllWindows()
cv2.imwrite("DoG_1.png", DoG_img)