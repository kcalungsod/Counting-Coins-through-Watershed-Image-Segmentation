#libraries/packages to be used
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# declaring figure 1 - shows original and processed image.
f1, (orig,processed) = plt.subplots(1,2)

img = cv.imread('water_coins.jpg')

orig.imshow(img)
orig.set_title("Original Image")

# image properties: array of original image
print("Image Properties")
print("-Array of the Original Image: " + str(img))
print('\n')

# gray-scale and otsu binarization thresholding
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)

# declaring figure 2 - shows grayscale and image with noise removal.
f2, (gr,noise) = plt.subplots(1,2)

gr.imshow(thresh)
gr.set_title("Grey tone")

noise.imshow(opening)
noise.set_title("Noise Removal")

# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)

# finding sure foreground area using distance transform
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# declaring figure 3 - shows background and distance transform.
f3, (back,dist) = plt.subplots(1,2)
back.imshow(sure_bg)
back.set_title("Sure Background")

dist.imshow(dist_transform)
dist.set_title("Distance Transform")

# finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)

# declaring figure 4 - shows foreground and unknown region.
f4, (fore,un) = plt.subplots(1,2)

fore.imshow(sure_fg)
fore.set_title("Sure Foreground - Threshold")

un.imshow(unknown)
un.set_title("Unknown Region")

# marker labelling
ret, markers = cv.connectedComponents(sure_fg)

# declaring figure 5 - shows markers and colormap.
f5, (mark,final) = plt.subplots(1,2)
mark.imshow(markers)
mark.set_title("Sure Coins")

# add one to all labels so that sure background is not 0, but 1
markers = markers+1
# now, mark the region of unknown with zero
markers[unknown==255] = 0

final.imshow(markers)
final.set_title("Jet Colormap")

# perform watershed and overlay markers to final processed result
markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,0]

# declaring figure 6 - shows watershed and final result.
f6, (ws,result) = plt.subplots(1,2)
ws.imshow(markers)
ws.set_title("Watershed")

result.imshow(img)
result.set_title("Final Output")

processed.imshow(img)
processed.set_title("After Processing")

# stores the sum of each marker in an array
area =[np.sum(markers==val) for val in range(ret)]

# image properties: size, dimensions, number of coins, array of processed image 
print("-Number of Pixels: " + str(img.size))
print("-Shape/Dimensions: " + str(img.shape))
print("-Number of Coins: " + str(len(area)-1))
print("-Array of the Processed Image: " + str(img))

plt.show()
