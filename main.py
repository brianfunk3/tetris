import numpy as np
from PIL import Image
import cv2
import time
import mss

"""
Testing space. Goal is to find an appropriate way to find a rectangle on screen and determine if it is a Tetris game
1. Find rectangles with OpenCV - need to get corner points
2. Abstract that space into a 2:1 ratio rectangle - might need to optimize points (or do this step within rectangle
   detection
3. Average pool into a 20x10 matrix.
4. Find error across clustering and use some threshold for making classification. Actually I might just want to get some
   error number to use and have classification made in main loop.
   
helpful? : https://stackoverflow.com/questions/7263621/how-to-find-corners-on-a-image-using-opencv
"""
#read in screen as a series of frames

def screen_record(timetrial = False):
    if timetrial:
        last_time = time.time()
    sct = mss.mss()
    while(True):
        sct_img = sct.grab(sct.monitors[1])
        printscreen = np.array(Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX'))
        # convert to grayscale
        gray = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
        if timetrial:
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()

        # show the screen
        """
        cv2.imshow('testing things', gray)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        """



screen_record(timetrial=True)