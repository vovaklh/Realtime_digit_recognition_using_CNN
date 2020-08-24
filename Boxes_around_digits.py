import cv2
import numpy as np
import imutils


def find_roi(image, rect):
    (startX, startY) = rect[0], rect[1]
    (endX, endY) = rect[0] + rect[2], rect[1] + rect[3]
    return image[startY:endY, startX:endX]


def extract_digits(image):
    # Convert to gray scale and apply Gaussian blur
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Threshold the image
    ret, img_th = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    cnts = cv2.findContours(img_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Get rectangles contains each contours
    rects = [cv2.boundingRect(ctr) for ctr in cnts]

    # Get roi
    digits = np.array([find_roi(img_th.copy(), rect) for rect in rects])

    return rects, digits
