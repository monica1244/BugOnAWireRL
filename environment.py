try:
    from PIL import Image
except ImportError:
    import Image
import numpy as np
from PIL import ImageGrab
# import pyscreenshot as ImageGrab
import cv2

import numpy as np
import time

import time
from pymouse import PyMouse
from pykeyboard import PyKeyboard

import os
import torch
from getkeys import key_check


# Set TOP_LEFT here as per the right pixel by checking mouse position
# when running this script
TOP_LEFT = (180, 420)
HEIGHT, WIDTH = 300, 600
SCREEN = (TOP_LEFT[0], TOP_LEFT[1], TOP_LEFT[0] + WIDTH, TOP_LEFT[1] + HEIGHT)
# BROWN in RGB
BROWN = [88, 52, 20]
WHITE = [255, 255, 255]
# GAME_OVER_PIXEL = (100, 10)
GAME_OVER_PIXEL = (8, 18)
NEW_GAME_PIXEL = (TOP_LEFT[0] + 295, TOP_LEFT[1] + 250)
FRAMES = 1
# 4 Actions: Nothing, UP, LEFT, RIGHT
N_ACTIONS = 4
RESOLUTION = 84

mouse = PyMouse()
keyboard = PyKeyboard()

def init_env():
    os.chdir("C:\Program Files\Mozilla Firefox")
    os.system('firefox.exe --new-tab "https://www.miniclip.com/games/bug-on-a-wire/en/bug.swf?mc_gamename=Bug+On+A+Wire&mc_hsname=1446&mc_iconBig=bugmedicon.jpg&mc_icon=bugsmallicon.jpg&mc_negativescore=0&mc_players_site=1&mc_scoreistime=0&mc_lowscore=0&mc_width=600&mc_height=300&mc_lang=en&mc_webmaster=0&mc_playerbutton=0&mc_v2=1&loggedin=0&mc_loggedin=0&mc_uid=0&mc_sessid=f78c2dbb92961726d9a87c8f9aa753d2&mc_shockwave=0&mc_gameUrl=%2Fgames%2Fbug-on-a-wire%2Fen%2F&mc_ua=705d28c&mc_geo=us-west-2&mc_geoCode=US&vid=0&vtype=ima&m_vid=1&mc_preroll_check=1&channel=miniclip.preroll&m_channel=miniclip.midroll&s_content=0&mc_plat_id=2&mc_extra=enable_personalized_ads%3D1&mc_image_cdn_path=https%3A%2F%2Favatars.miniclip.com%2F&login_allowed=1&dfp_video_url=https%253A%252F%252Fpubads.g.doubleclick.net%252Fgampad%252Fads%253Fsz%253D600x400%2526iu%253D%252F116850162%252FMiniclip.com_Preroll%2526ciu_szs%2526impl%253Ds%2526gdfp_req%253D1%2526env%253Dvp%2526output%253Dxml_vast2%2526unviewed_position_start%253D1%2526cust_params%253D%2526npa%253D0%2526cust_params%253DgCat%25253Dcategory_13%252526gName%25253Dgame_1446%252526width%25253D600%252526height%25253D300%252526page_domain%25253Dgames%252526gAATF%25253Dgaatf_Y%252526gLanguage%25253Dlanguage_en%252526gPageType%25253Dpagetype_gamepage%252526gDemo1%25253Ddemo1_1%252526gDemo2%25253Ddemo2_2%252526gPageUrl%25253Dhttps%2525253A%2525252F%2525252Fwww.miniclip.com%2525252Fgames%2525252Fbug-on-a-wire%2525252Fen%2525252F%2526url%253D&fn=bug.swf"')
    time.sleep(10)
    os.chdir(r"C:\Users\nihal\github\bugOnAWireRL")


def generate_state(image_arrays, dim=RESOLUTION, frames=1, channels=3):
    state = np.zeros((frames, channels, dim, dim))
    for i, img in enumerate(image_arrays):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Color handling
        brown_lo = np.array([0, 50, 100])
        brown_hi = np.array([0, 100, 200])
        mask = cv2.inRange(img, brown_lo, brown_hi)
        img[mask>0] = (0, 0, 0)
        blue_lo = np.array([180, 140, 20])
        blue_hi = np.array([255, 220, 120])
        mask = cv2.inRange(img, blue_lo, blue_hi)
        img[mask>0] = (0, 0, 0)
        crow_lo = np.array([40, 40, 40])
        crow_hi = np.array([60, 60, 60])
        mask = cv2.inRange(img, crow_lo, crow_hi)
        img[mask>0] = (0, 0, 255)
        # Remove corners
        img[:40, :200, :] = 0
        img[:40, -200:, :] = 0
        img = cv2.resize(img,(dim, dim)) / 255.0
        # if np.random.random_sample() > 0.5:
        #     mouse.click(TOP_LEFT[0] + 10, TOP_LEFT[1] + 10, 1)
        #     cv2.imshow('img', img)
        #     cv2.waitKey(0)

        state[i] = np.transpose(img, (2, 0, 1))
    return state

def start_new_game():
    mouse.click(TOP_LEFT[0], TOP_LEFT[1], 1)
    mouse.click(NEW_GAME_PIXEL[0], NEW_GAME_PIXEL[1], 1)
    image_arrays = []
    for _ in range(FRAMES):
        image_arrays.append(np.array(ImageGrab.grab(bbox=SCREEN)))
        time.sleep(0.1)
    return generate_state(image_arrays)

def is_game_over(frame):
    '''
    Checks pixel GAME_OVER_PIXEL corresponding to a patch in the scoreboard
    that turns from white to brown upon game over screen transition.

    If that pixel has value != WHITE
    return True i.e. game is over
    '''
    return list(frame[GAME_OVER_PIXEL]) != WHITE


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

    state = np.array(ImageGrab.grab(bbox=SCREEN))
    reward = sum(get_crows_positions(state)) * 1
    if is_game_over(state):
        state = None
        frame_diff = None
        reward = -5
    else:
        image_arrays = []
        for _ in range(FRAMES):
            frame_i = np.array(ImageGrab.grab(bbox=SCREEN))
            # time.sleep(0.2)
            # frame_f = np.array(ImageGrab.grab(bbox=SCREEN))
            # frame_diff = np.absolute(frame_f - frame_i)
            frame_diff = frame_i
            image_arrays.append(frame_diff)
        state = generate_state(image_arrays)
    return reward, state


def get_action_reward_and_next_state():
    action = key_check()
    state = np.array(ImageGrab.grab(bbox=SCREEN))
    reward = sum(get_crows_positions(state)) * 1
    if is_game_over(state):
        state = None
        frame_diff = None
        reward = -5
    else:
        image_arrays = []
        for _ in range(FRAMES):
            frame_i = np.array(ImageGrab.grab(bbox=SCREEN))
            time.sleep(0.1)
            # frame_f = np.array(ImageGrab.grab(bbox=SCREEN))
            # frame_diff = np.absolute(frame_f - frame_i)
            frame_diff = frame_i
            image_arrays.append(frame_diff)
        state = generate_state(image_arrays)
    return action, reward, state


def get_crows_positions(img):
    '''
    Takes img as the unprocessed screen capture
    Returns whether crows exist at the bug's line
    in a boolean array like
    [0 1 0 0] means bug only at wire_1
    '''
    crow_lo = np.array([190, 190, 0])
    crow_hi = np.array([250, 250, 0])

    wire_0 = img[630-420:, 220-180:360-180, :]
    wire_1 = img[630-420:, 380-180:480-180, :]
    wire_2 = img[630-420:, 500-180:600-180, :]
    wire_3 = img[630-420:, 605-180:700-180, :]


    wire = img[680-420:, 295-180:660-180, :]

    mask_0 = cv2.inRange(wire_0, crow_lo, crow_hi)
    mask_1 = cv2.inRange(wire_1, crow_lo, crow_hi)
    mask_2 = cv2.inRange(wire_2, crow_lo, crow_hi)
    mask_3 = cv2.inRange(wire_3, crow_lo, crow_hi)

    crows = [0, 0, 0, 0]
    if np.sum(np.nonzero(mask_0)) > 0:
        crows[0] = 1
    if np.sum(np.nonzero(mask_1)) > 0:
        crows[1] = 1
    if np.sum(np.nonzero(mask_2)) > 0:
        crows[2] = 1
    if np.sum(np.nonzero(mask_3)) > 0:
        crows[3] = 1

    # cv2.imshow('window', wire_3)
    return crows

def get_bug_position(img):
    '''
    Takes img as the unprocessed screen capture
    Returns bug's position as wire number
    0 through 3 from Left to Right
    '''
    green_lo = np.array([0, 90, 0])
    green_hi = np.array([0, 150, 0])

    wire_0 = img[615-420:700-420, 295-180:340-180, :]
    wire_1 = img[615-420:700-420, 405-180:450-180, :]
    wire_2 = img[615-420:700-420, 505-180:550-180, :]
    wire_3 = img[615-420:700-420, 615-180:660-180, :]


    mask_0 = cv2.inRange(wire_0, green_lo, green_hi)
    mask_1 = cv2.inRange(wire_1, green_lo, green_hi)
    mask_2 = cv2.inRange(wire_2, green_lo, green_hi)
    mask_3 = cv2.inRange(wire_3, green_lo, green_hi)

    position = -1
    if np.sum(np.nonzero(mask_0)) > 0:
        position = 0
    elif np.sum(np.nonzero(mask_1)) > 0:
        position = 1
    elif np.sum(np.nonzero(mask_2)) > 0:
        position = 2
    elif np.sum(np.nonzero(mask_3)) > 0:
        position = 3
    # cv2.imshow('window', mask_0)
    return position


def screen_record():
    last_time = time.time()
    while(True):
        img = np.array(ImageGrab.grab(bbox=SCREEN))
        printscreen = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
        # print('loop took {} seconds'.format(time.time() - last_time))
        m_pos = mouse.position()
        color = img[(np.clip(m_pos[1] - TOP_LEFT[1], 0, HEIGHT - 1),
            np.clip(m_pos[0] - TOP_LEFT[0], 0, WIDTH - 1))]

        # Color handling
        brown_lo = np.array([0, 50, 100])
        brown_hi = np.array([0, 100, 200])
        mask = cv2.inRange(printscreen, brown_lo, brown_hi)
        printscreen[mask>0] = (0, 0, 0)
        blue_lo = np.array([180, 140, 20])
        blue_hi = np.array([255, 220, 120])
        mask = cv2.inRange(printscreen, blue_lo, blue_hi)
        printscreen[mask>0] = (0, 0, 0)
        crow_lo = np.array([40, 40, 40])
        crow_hi = np.array([60, 60, 60])
        mask = cv2.inRange(printscreen, crow_lo, crow_hi)
        printscreen[mask>0] = (0, 0, 255)
        # Remove corners
        printscreen[:40, :200, :] = 0
        printscreen[:40, -200:, :] = 0

        # print("Mouse position: {} Color: {}".format(m_pos, color))
        # print("Game Over? " + str(is_game_over(img)))
        last_time = time.time()

        # print("Bug position: {}".format(get_bug_position(img)))
        r = sum(get_crows_positions(img))
        if r > 0 and not is_game_over(img):
            print("Crow positions: {}".format(get_crows_positions(img)))

        printscreen = cv2.resize(printscreen,(84, 84)) / 255.0

        # cv2.imshow('window', printscreen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    screen_record()
