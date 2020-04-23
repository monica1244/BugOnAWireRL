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
import pymouse
from pymouse import PyMouse

import os


# Set TOP_LEFT here as per the right pixel by checking mouse position
# when running this script
TOP_LEFT = (630, 330)
HEIGHT, WIDTH = 450, 650
SCREEN = (TOP_LEFT[0], TOP_LEFT[1], TOP_LEFT[0] + WIDTH, TOP_LEFT[1] + HEIGHT)
BROWN = [88, 52, 20]
GAME_OVER_PIXEL = (150, 50)
NEW_GAME_PIXEL = (TOP_LEFT[0] + 325, TOP_LEFT[1] + 345)

mouse = pymouse.PyMouse()
keyboard = PyKeyboard()

def init_env():
    os.chdir("C:\Program Files\Mozilla Firefox")
    os.system('firefox.exe --new-tab "https://www.miniclip.com/games/bug-on-a-wire/en/bug.swf?mc_gamename=Bug+On+A+Wire&mc_hsname=1446&mc_iconBig=bugmedicon.jpg&mc_icon=bugsmallicon.jpg&mc_negativescore=0&mc_players_site=1&mc_scoreistime=0&mc_lowscore=0&mc_width=600&mc_height=300&mc_lang=en&mc_webmaster=0&mc_playerbutton=0&mc_v2=1&loggedin=0&mc_loggedin=0&mc_uid=0&mc_sessid=f78c2dbb92961726d9a87c8f9aa753d2&mc_shockwave=0&mc_gameUrl=%2Fgames%2Fbug-on-a-wire%2Fen%2F&mc_ua=705d28c&mc_geo=us-west-2&mc_geoCode=US&vid=0&vtype=ima&m_vid=1&mc_preroll_check=1&channel=miniclip.preroll&m_channel=miniclip.midroll&s_content=0&mc_plat_id=2&mc_extra=enable_personalized_ads%3D1&mc_image_cdn_path=https%3A%2F%2Favatars.miniclip.com%2F&login_allowed=1&dfp_video_url=https%253A%252F%252Fpubads.g.doubleclick.net%252Fgampad%252Fads%253Fsz%253D600x400%2526iu%253D%252F116850162%252FMiniclip.com_Preroll%2526ciu_szs%2526impl%253Ds%2526gdfp_req%253D1%2526env%253Dvp%2526output%253Dxml_vast2%2526unviewed_position_start%253D1%2526cust_params%253D%2526npa%253D0%2526cust_params%253DgCat%25253Dcategory_13%252526gName%25253Dgame_1446%252526width%25253D600%252526height%25253D300%252526page_domain%25253Dgames%252526gAATF%25253Dgaatf_Y%252526gLanguage%25253Dlanguage_en%252526gPageType%25253Dpagetype_gamepage%252526gDemo1%25253Ddemo1_1%252526gDemo2%25253Ddemo2_2%252526gPageUrl%25253Dhttps%2525253A%2525252F%2525252Fwww.miniclip.com%2525252Fgames%2525252Fbug-on-a-wire%2525252Fen%2525252F%2526url%253D&fn=bug.swf"')
    time.sleep(5)


def start_new_game():
    mouse.click(NEW_GAME_PIXEL[0], NEW_GAME_PIXEL[1], 1)
    state = np.array(ImageGrab.grab(bbox=SCREEN))
    return state

def is_game_over(frame):
    '''
    Checks pixel [150, 50] corresponding to a patch in the sky that
    turns from blue to brown upon game over screen transition.

    If that pixel has value == BROWN
    return True i.e. game is over

    Note: This has a small lag, since the game over screen is presented
    after the crow eating animation
    '''
    return list(frame[GAME_OVER_PIXEL]) == BROWN


def press_key_with_wait(key):
    keyboard.press_key(key)
    time.sleep(0.1)
    keyboard.release_key(key)

def get_reward_and_next_state(action):
    if action == 0:
        # NO ACTION
        time.sleep(0.1)
    elif action == 1:
        # UP KEY
        press_key_with_wait(keyboard.up_key)
    elif action == 2:
        # RIGHT KEY
        press_key_with_wait(keyboard.right_key)
    elif action == 3:
        # LEFT KEY
        press_key_with_wait(keyboard.left_key)
    else:
        raise Exception

    if not is_game_over():
        state = np.array(ImageGrab.grab(bbox=SCREEN))
    else:
        state = None
    return state


def screen_record():
    last_time = time.time()
    while(True):
        printscreen =  np.array(ImageGrab.grab(bbox=SCREEN))
        print('loop took {} seconds'.format(time.time() - last_time))
        print("Mouse position: {}".format(mouse.position()))
        print("Game Over? " + str(is_game_over(printscreen)))
        last_time = time.time()
        cv2.imshow('window',cv2.cvtColor(printscreen,cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    screen_record()
