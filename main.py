import numpy as np
# from PIL import ImageGrab
import pyscreenshot as ImageGrab
import cv2
import time
from pymouse import PyMouse
from pykeyboard import PyKeyboard

# def screen_record(): 
#     last_time = time.time()
#     while(True):
#         # 800x600 windowed mode for GTA 5, at the top left position of your main screen.
#         # 40 px accounts for title bar. 
#         printscreen =  np.array(ImageGrab.grab(bbox=(100,200,750,650)))
#         print('loop took {} seconds'.format(time.time()-last_time))
#         last_time = time.time()
#         cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()
#             break

# screen_record()

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