#!/usr/bin/env python3

# importing needed packages 
import cv2
import pdb
import numpy as np
import matplotlib.pyplot as plt

path = '/Users/parth/coding/AvBotz22-23/part1/assets/'

# reading the image the image
# if the image isnt there then it will say "Error: File not found"
image = cv2.imread(path + '4.png')
if image is None:
    print("Error: File not found")
    exit(0)

plt.imshow(image)
plt.show()

# Converts from BGR TO RGB and then shows the image
# I did this b/c CV2 uses BGR automatically and the image we are reading for an input is in RGB
# After this I converted it to HSV so I can input the needed HSV vals to iso the orange PM :)
# I also displayed the image using imshow()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
plt.imshow(image)
plt.show()

# masking values to iso the orange PM
# https://realpython.com/python-opencv-color-spaces/
light_orange = np.array([0, 110, 130])
dark_orange = np.array([240, 150, 230])

mask = cv2.inRange(hsv_image, light_orange, dark_orange)
result = cv2.bitwise_and(image, image, mask=mask)
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()

# Grayscaling the image 
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
plt.show()

# Find Canny edges
edged = cv2.Canny(gray, 30, 200)

contours, hierarchy = cv2.findContours(edged,
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
plt.imshow(edged)
plt.show()
print("Number of Contours found = " + str(len(contours)))

# getting the (x,y) center of the countour https://www.geeksforgeeks.org/python-opencv-find-center-of-contour/
for i in contours:
    M = cv2.moments(i)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    print(f"x: {cx} y: {cy}")
   # https://theailearner.com/tag/angle-of-rotation-by-cv2-minarearect/
   # Im using the minAreaReact() to get the angle of the PM
   # the minAreaReact() O/P 3 values: the (x,y) center (in this case im getting the (x,y) of PM)
   # height and width (in this case its the height and width of the PM), 
   # and last it outputs the angle orientation (in this case its the angle orientation of pm)
   # I then used the boxPoints() method to make a box around PM (goes clockwise)
   # It draws a box using 4 points that have been converted into integers 
   # the angle is the angle between 2 points and a line that lies outside those points 
    rect = cv2.minAreaRect(i)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    rounded_angle = round(rect[2])
    print(rounded_angle)