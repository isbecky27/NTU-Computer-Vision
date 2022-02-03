import cv2
import math
import numpy as np
image = cv2.imread('lena.bmp', 0)
img_padding = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)
h, w = image.shape

def mask(img, i, j, mask):
    rows, cols = mask.shape
    value = 0
    for a in range(rows):
        for b in range(cols):
            value += img[i+a][j+b] * mask[a][b]
    return value

def edgeDetector(img, threshold, mask1, mask2, start = 0):
    output = np.zeros((h, w), np.uint8)
    img_padding = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    for i in range(h):
        for j in range(w):
            gradient = math.sqrt(mask(img_padding, i+start, j+start, mask1)**2 + mask(img_padding, i+start, j+start, mask2)**2)
            if gradient >= threshold:
                output[i][j] = 0
            else: 
                output[i][j] = 255
    return output

# Robert's Operator
robert_1 = np.array([[-1, 0],
                    [0, 1]])

robert_2 = np.array([[0, -1],
                    [1, 0]])

robert_img = edgeDetector(image, 30, robert_1, robert_2, 1)
cv2.imshow("Robert's Operator", robert_img)
cv2.waitKey(0)                             
cv2.destroyAllWindows()
cv2.imwrite("Robert_30.png", robert_img)

# Prewitt's Edge Detector
prewitt_1 = np.array([[-1, -1, -1],
                    [ 0,  0,  0],
                    [ 1,  1,  1]])

prewitt_2 = np.array([[-1,  0,  1],
                    [-1,  0,  1],
                    [-1,  0,  1]])

prewitt_img = edgeDetector(image, 24, prewitt_1, prewitt_2)
cv2.imshow("Prewitt's Edge Detector", prewitt_img)
cv2.waitKey(0)                             
cv2.destroyAllWindows()
cv2.imwrite("Prewitt_24.png", prewitt_img)

# Sobel's Edge Detector
sobel_1 = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

sobel_2 = np.array([[-1,  0,  1],
                    [-2,  0,  2],
                    [-1,  0,  1]])

sobel_img = edgeDetector(image, 38, sobel_1, sobel_2)
cv2.imshow("Sobel's Edge Detector", sobel_img)
cv2.waitKey(0)                             
cv2.destroyAllWindows()
cv2.imwrite("Sobel_38.png", sobel_img)

# Frei and Chen's Gradient Operator
frei_1 = np.array([[-1, -1*math.sqrt(2), -1],
                    [ 0,  0,  0],
                    [ 1,  math.sqrt(2),  1]])

frei_2 = np.array([[-1,  0,  1],
                    [-1*math.sqrt(2),  0,  math.sqrt(2)],
                    [-1,  0,  1]])

frei_img = edgeDetector(image, 30, frei_1, frei_2)
cv2.imshow("Frei and Chen's Gradient Operator", frei_img)
cv2.waitKey(0)                             
cv2.destroyAllWindows()
cv2.imwrite("Frei_30.png", frei_img)

def compassOperator(img, threshold, masks, padding = 1):
    output = np.zeros((h, w), np.uint8)
    img_padding = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REFLECT)
    for i in range(h):
        for j in range(w):
            maskValue = []
            for k in range(len(masks)):
                maskValue.append(mask(img_padding, i, j, masks[k]))
            if max(maskValue) >= threshold:
                output[i][j] = 0
            else: 
                output[i][j] = 255
    return output

# Kirsch's Compass Operator
k_0 = np.array([[-3, -3,  5], [-3,  0,  5], [-3, -3,  5]])
k_1 = np.array([[-3,  5,  5], [-3,  0,  5], [-3, -3, -3]])
k_2 = np.array([[ 5,  5,  5], [-3,  0, -3], [-3, -3, -3]])
k_3 = np.array([[ 5,  5, -3], [ 5,  0, -3], [-3, -3, -3]])
k_4 = np.array([[ 5, -3, -3], [ 5,  0, -3], [ 5, -3, -3]])
k_5 = np.array([[-3, -3, -3], [ 5,  0, -3], [ 5,  5, -3]])
k_6 = np.array([[-3, -3, -3], [-3,  0, -3], [ 5,  5,  5]])
k_7 = np.array([[-3, -3, -3], [-3,  0,  5], [-3,  5,  5]])
kirsches = [k_0, k_1, k_2, k_3, k_4, k_5, k_6, k_7]

kirsch_img = compassOperator(image, 135, kirsches)
cv2.imshow("Kirsch's Compass Operator", kirsch_img)
cv2.waitKey(0)                             
cv2.destroyAllWindows()
cv2.imwrite("Kirsch_135.png", kirsch_img)

# Robinson's Compass Operator
r_0 = np.array([[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]])
r_1 = np.array([[ 0,  1,  2], [-1,  0,  1], [-2, -1,  0]])
r_2 = np.array([[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]])
r_3 = np.array([[ 2,  1,  0], [ 1,  0, -1], [ 0, -1, -2]])
r_4 = np.array([[ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]])
r_5 = np.array([[ 0, -1, -2], [ 1,  0, -1], [ 2,  1,  0]])
r_6 = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]])
r_7 = np.array([[-2, -1,  0], [-1,  0,  1], [ 0,  1,  2]])
robinsons = [r_0, r_1, r_2, r_3, r_4, r_5, r_6, r_7]

robinson_img = compassOperator(image, 43, robinsons)
cv2.imshow("Robinson's Compass Operator", robinson_img)
cv2.waitKey(0)                             
cv2.destroyAllWindows()
cv2.imwrite("Robinson_43.png", robinson_img)

# Nevatia-Babu 5x5 Operator
n_0 = np.array([[ 100, 100, 100, 100, 100],
                [ 100, 100, 100, 100, 100],
                [   0,   0,   0,   0,   0],
                [-100,-100,-100,-100,-100],
                [-100,-100,-100,-100,-100]])

n_30 = np.array([[ 100, 100, 100, 100, 100],
                 [ 100, 100, 100,  78, -32],
                 [ 100,  92,   0, -92,-100],
                 [  32, -78,-100,-100,-100],
                 [-100,-100,-100,-100,-100]])

n_60 = np.array([[ 100, 100, 100,  32,-100],
                 [ 100, 100,  92, -78,-100],
                 [ 100, 100,   0,-100,-100],
                 [ 100,  78, -92,-100,-100],
                 [ 100, -32,-100,-100,-100]])

n_270 = np.array([[-100,-100,   0, 100, 100],
                  [-100,-100,   0, 100, 100],
                  [-100,-100,   0, 100, 100],
                  [-100,-100,   0, 100, 100],
                  [-100,-100,   0, 100, 100]])

n_300 = np.array([[-100,  32, 100, 100, 100],
                  [-100, -78,  92, 100, 100],
                  [-100,-100,   0, 100, 100],
                  [-100,-100, -92,  78, 100],
                  [-100,-100,-100, -32, 100]])

n_330 = np.array([[ 100, 100, 100, 100, 100],
                  [ -32,  78, 100, 100, 100],
                  [-100, -92,   0,  92, 100],
                  [-100,-100,-100, -78,  32],
                  [-100,-100,-100,-100,-100]])
nevatias = [n_0, n_30, n_60, n_270, n_300, n_330]

nevatia_img = compassOperator(image, 12500, nevatias, 2)
cv2.imshow("Nevatia-Babu 5x5 Operator", nevatia_img)
cv2.waitKey(0)                             
cv2.destroyAllWindows()
cv2.imwrite("Nevatia_12500.png", nevatia_img)