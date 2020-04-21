try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import numpy as np
from PIL import ImageGrab
# import pyscreenshot as ImageGrab
import cv2

import numpy as np
import time

import time


def is_game_over(frame):
    '''
    Checks pixel [150, 50] corresponding to a patch in the sky that
    turns from blue to brown upon game over screen transition.

    If that pixel has value == [88, 52, 20] (brown)
    return True i.e. game is over

    Note: This has a small lag, since the game over screen is presented
    after the crow eating animation
    '''
    return list(frame[150,50]) == [88, 52, 20]

def screen_record():
    last_time = time.time()
    while(True):
        printscreen =  np.array(ImageGrab.grab(bbox=(100,200,750,650)))
        print('loop took {} seconds'.format(time.time()-last_time))
        print("Game Over? " + str(is_game_over(printscreen)))
        last_time = time.time()
        cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

screen_record()
