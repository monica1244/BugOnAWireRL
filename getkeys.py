# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

import win32api as wapi
import win32con as wcon
import time

def key_check():
    if wapi.GetAsyncKeyState(wcon.VK_UP):
        return 1
    if wapi.GetAsyncKeyState(wcon.VK_RIGHT):
        return 2
    if wapi.GetAsyncKeyState(wcon.VK_LEFT):
        return 3
    else:
        return 0

if __name__ == '__main__':
    while(True):
        time.sleep(0.2)
        print(key_check())
