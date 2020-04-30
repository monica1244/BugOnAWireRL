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
from getkeys import key_check_flappy


# Set TOP_LEFT here as per the right pixel by checking mouse position
# when running this script
TOP_LEFT = (107, 197)
HEIGHT, WIDTH = 639, 479
SCREEN = (TOP_LEFT[0], TOP_LEFT[1], TOP_LEFT[0] + WIDTH, TOP_LEFT[1] + HEIGHT)
# BROWN in RGB
BROWN = [232, 97, 1]
WHITE = [255, 255, 255]
GAME_OVER_PIXEL = (567 - TOP_LEFT[1], 216 - TOP_LEFT[0])
NEW_GAME_PIXEL = (TOP_LEFT[0] + 110, TOP_LEFT[1] + 370)
FRAMES = 1
# 4 Actions: Nothing, UP, LEFT, RIGHT
N_ACTIONS = 2
RESOLUTION = 84

mouse = PyMouse()
keyboard = PyKeyboard()

def init_env():
    os.chdir("C:\Program Files\Mozilla Firefox")
    os.system('firefox.exe --new-tab "https://flappybird.io/"')
    time.sleep(10)
    os.chdir(r"C:\Users\nihal\github\bugOnAWireRL")


def generate_state(frame, dim=RESOLUTION, channels=1):
    state = cv2.resize(frame, (dim, dim)) / 255.0
    # if np.random.random_sample() > 0.5:
    #     mouse.click(TOP_LEFT[0] + 10, TOP_LEFT[1] + 10, 1)
    #     cv2.imshow('img', state)
    #     cv2.moveWindow('img', 1400, 30)
    #     cv2.waitKey(0)
    return state[None, :, :].astype(np.float32)

def start_new_game():
    mouse.click(NEW_GAME_PIXEL[0], NEW_GAME_PIXEL[1], 1)
    time.sleep(0.1)
    mouse.click(NEW_GAME_PIXEL[0], NEW_GAME_PIXEL[1], 1)
    frame = pre_process(np.array(ImageGrab.grab(bbox=SCREEN)))
    state = generate_state(frame)
    return state

def is_game_over(frame, pre_processed_frame):
    '''
    Checks pixel GAME_OVER_PIXEL corresponding to a patch in the
    restart button that is BROWN

    If that pixel has value == BROWN
    return True i.e. game is over
    '''
    if list(frame[GAME_OVER_PIXEL]) == BROWN or np.sum(pre_processed_frame) == 0:
        return True
    else:
        return False


def pre_process(img):
    ## Filter Bird
    mask = get_mask_for_colors(img, [[234, 80, 64],
                                            [212, 191, 39],
                                            [235, 252, 221],
                                            [221, 226, 177],
                                            [228, 129, 22]])

    ## Filter Pipes
    mask += get_mask_for_colors(img, [[85, 128, 34],
                                              [228, 253, 139],
                                              [155, 227, 89],
                                              [115, 191, 46],
                                              [157, 230, 90],
                                              [115, 189, 46],
                                              [117, 192, 47]])
    return mask


def get_reward_and_next_state(action, dim=84):
    if action == 0:
        # NO ACTION
        # time.sleep(0.1)
        pass
    elif action == 1:
        # mouse click
        mouse.click(TOP_LEFT[0], TOP_LEFT[1], 1)
        # time.sleep(0.1)

    frame = np.array(ImageGrab.grab(bbox=SCREEN))
    pre_processed_frame = pre_process(frame)
    if is_game_over(frame, pre_processed_frame):
        state = None
        frame_diff = None
        reward = -2
    else:
        reward = pipe_reward(pre_processed_frame) + bird_in_view_reward(frame) + 0.1
        state = generate_state(pre_processed_frame)
    return reward, state


def get_action_reward_and_next_state(action, dim=84):

    action = key_check_flappy()
    frame = np.array(ImageGrab.grab(bbox=SCREEN))
    pre_processed_frame = pre_process(frame)
    if is_game_over(frame, pre_processed_frame):
        state = None
        frame_diff = None
        reward = -2
    else:
        reward = pipe_reward(pre_processed_frame) + bird_in_view_reward(frame) + 0.1
        state = generate_state(pre_processed_frame)
    return action,reward, state




def pipe_reward(pre_processed_frame):
    # wire = pre_processed_frame[:, 305-TOP_LEFT[0]:340-TOP_LEFT[0]]
    # cv2.imshow('img', wire)
    if pre_processed_frame[(750 - TOP_LEFT[1], 340 - TOP_LEFT[0])] > 0 and \
            pre_processed_frame[(205 - TOP_LEFT[1], 340 - TOP_LEFT[0])] > 0:
        return 1
    else:
        return 0

def bird_in_view_reward(frame):
    mask = get_mask_for_colors(frame, [[234, 80, 64],
                                        [212, 191, 39],
                                        [235, 252, 221],
                                        [221, 226, 177],
                                        [228, 129, 22]])
    if np.sum(mask) > 0:
        return 0
    else:
        return -1


def get_mask_for_colors(img, colors):
    '''
    Takes RGB colors and returns a mask filtering those colors.
    '''
    mask = np.zeros((img.shape[0], img.shape[1]))
    for color in colors:
        color = np.array(color)
        color_lo = color - np.array([1, 1, 1])
        color_hi = color + np.array([1, 1, 1])
        mask += cv2.inRange(img, color_lo, color_hi)
    return mask



def screen_record():
    last_time = time.time()
    # start_new_game()
    while(True):
        printscreen = np.array(ImageGrab.grab(bbox=SCREEN))
        print('loop took {:.2f} seconds'.format(time.time() - last_time))
        last_time = time.time()

        m_pos = mouse.position()
        color = printscreen[(np.clip(m_pos[1] - TOP_LEFT[1], 0, HEIGHT - 1),
            np.clip(m_pos[0] - TOP_LEFT[0], 0, WIDTH - 1))]

        pre_processed_frame = pre_process(printscreen)
        state = generate_state(pre_processed_frame)
        # print("Mouse position: {} Color: {}".format(m_pos, color))
        # print("Game Over? " + str(is_game_over(img, pre_processed_frame)))

        ## PLAY with random action
        # get_reward_and_next_state(np.random.randint(0, 2))
        # if is_game_over(pre_processed_frame):
        #     break

        cv2.imshow('window', pre_processed_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    screen_record()
