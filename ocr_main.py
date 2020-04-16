try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import numpy as np
import pyscreenshot as ImageGrab
import cv2

import numpy as np
import time
import pyscreenshot as ImageGrab

import time

whole = (200,390,640,425)
hours = (200,390,230,425)
minutes = (280,390,305,425)
seconds = (370,390,400,425)

def ocr():
    img = np.array(ImageGrab.grab(bbox=whole))

    ## Change Color scale
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## Thresholding
    # thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    # img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Filter White
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    lower_white = np.array([0,0,250], dtype=np.uint8)
    upper_white = np.array([0,0,255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask=mask)

    cv2.imshow('frame',img)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    print(pytesseract.image_to_string(res))
    cv2.imshow('window', res)
    cv2.imwrite('white.png', res)
    # time.sleep(1)

def screen_record(): 
    while(True):
        ocr()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

# screen_record()
ocr()

# Simple image to string
# img = Image.open('img.png')
# print(pytesseract.image_to_string(img))
