import numpy as np
import cv2
import math
import random

flagList = [0]

def find_root(index):
    root = index
    while flagList[root] != root:
        root = flagList[root]
    return root

def union(index1, index2):
    root1 = find_root(index1)
    root2 = find_root(index2)
    
    if root1 > root2:
        flagList[root1] = root2
    elif root1 < root2:
        flagList[root2] = root1

if __name__ == '__main__':

    img = cv2.imread('a.jpg', 0)
    h, w = img.shape
    label = [[0 for i in range(w)] for j in range(h)] 
    # label = np.zeros((h, w), np.uint8) # error
      
    flag = 1

    # first scan
    for i in range(h):
        for j in range(w):

            if img[i][j] < 128:
                continue

            # first pixel
            if i == 0 and j == 0:
                label[i][j] = flag
                flagList.append(flag)
                flag += 1

            # judge left component
            elif i == 0 and j > 0:
                left = label[i][j-1]

                if left != 0:
                    label[i][j] = left
                else:
                    label[i][j] = flag
                    flagList.append(flag)
                    flag += 1
                
            # judge upper component
            elif j == 0 and i > 0:
                upper = label[i-1][j]

                if upper != 0:
                    label[i][j] = upper
                else:
                    label[i][j] = flag
                    flagList.append(flag)
                    flag += 1
                    
            # judge left and upper component
            else:
                left = label[i][j-1]
                upper = label[i-1][j]

                if left == 0 and upper == 0:
                    label[i][j] = flag
                    flagList.append(flag)
                    flag += 1
                elif left == 0:
                    label[i][j] = upper
                elif upper == 0:
                    label[i][j] = left
                else:
                    label[i][j] = min(left, upper)
                    union(left, upper)
                    
    # second scan
    for i in range(h):
        for j in range(w):
            if label[i][j] != 0:
                label[i][j] = find_root(label[i][j])

    # omit regions that have a pixel count less than 500
    unique, counts = np.unique(label, return_counts=True)
    labelCounts = dict(zip(unique, counts))
    for key, value in list(labelCounts.items()):
        if value < 500 or key == 0:
            del labelCounts[key]
    
    # draw bounding box and regions with + at centroid
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for key in labelCounts.keys():
        indexX = [i for i in range(len(label[0])) for j in range(len(label)) if label[j][i] == key]
        indexY = [i for i, value in enumerate(label) for j in value if j == key]
        x1, y1, x2, y2 = min(indexX), min(indexY), max(indexX), max(indexY)
        centroidX = round(sum(indexX)/len(indexX))
        centroidY = round(sum(indexY)/len(indexY))
        # print(x1, y1, x2, y2)
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(output, (centroidX, centroidY), 4, (0, 0, 255))

    cv2.imshow('Result', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("c.jpg", output)  