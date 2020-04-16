import numpy as np
# from PIL import ImageGrab
import pyscreenshot as ImageGrab
import cv2
import time
from pymouse import PyMouse
from pykeyboard import PyKeyboard


for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

m = PyMouse()
k = PyKeyboard()

m.click(400, 520, 1)

k.press_key(k.right_key)
for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)
k.release_key(k.right_key)

k.press_key(k.left_key)
for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)
k.release_key(k.left_key)

k.press_key(k.right_key)
time.sleep(0.1)
k.release_key(k.right_key)
time.sleep(0.1)
k.press_key(k.right_key)
time.sleep(0.1)
k.release_key(k.right_key)

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)
k.press_key(k.up_key)