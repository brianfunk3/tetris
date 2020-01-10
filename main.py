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
3. Average pool into a 20x10 matrix. OR std dev pooling to find the consistency in each square? Do a threshhold? Hmmm
4. Find error across clustering and use some threshold for making classification. Actually I might just want to get some
   error number to use and have classification made in main loop.
   
helpful? : https://stackoverflow.com/questions/7263621/how-to-find-corners-on-a-image-using-opencv
"""
ratio_error = .01

def tetrisy(c):
    return True
    """
    return (1-ratio_error) <= (c[0][0]/c[3][0]) <= (1+ratio_error) and (1-ratio_error) <= (c[1][0]/c[2][0]) <= (1+ratio_error) \
        and (1-ratio_error) <= (c[0][1]/c[3][1]) <= (1+ratio_error) and (1-ratio_error) <= (c[1][1]/c[2][1]) <= (1+ratio_error) #\
        #and abs(c[1][1]-c[2][1])*(1-ratio_error) <= abs(c[0][0]-c[1][0])*2 <= abs(c[1][1]-c[2][1])*(1+ratio_error)
    """

def screen_record(timetrial = False, capture='screen', show=False):
    if timetrial:
        last_time = time.time()
    if capture == 'screen':
        sct = mss.mss()
    elif capture == 'cam':
        camera = cv2.VideoCapture(0)

    while(True):
        if capture == 'screen':
            monitor = {"top": 0, "left": 0, "width": 950, "height": 1080}
            sct_img = sct.grab(monitor)
            printscreen = np.array(sct_img)
        elif capture == 'cam':
            printscreen = camera.read()[1]
        # convert image to grayscale
        gray = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
        # smooth the image
        #smooth = cv2.GaussianBlur(gray,(5,5),0)
        # threshold the image - actually a bad idea lol
        #thresh = cv2.threshold(gray,160,255,cv2.THRESH_BINARY)[1]
        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
        # edge detection
        edges = cv2.Canny(thresh,100,200)
        # find contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # approximate the polys (if existent)?
        if len(contours) > 0:
            for c in contours:
                # uh
                peri = cv2.arcLength(c, True)
                poly = cv2.approxPolyDP(c,.04 * peri,True)
                if len(poly) == 4 and cv2.contourArea(poly) > 3000:
                    # polygon has 4 vertices, now need to check if it is tetris-y?
                    poly = poly.reshape(-1,2)
                    if tetrisy(poly):
                        #print("hmm")
                        cv2.drawContours(printscreen, [c], -1, (0, 255, 0), 3)

        if timetrial:
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()

        # show the screen
        if show:
            cv2.imshow('testing things', thresh)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


screen_record(show=True)