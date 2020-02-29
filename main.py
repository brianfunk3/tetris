import numpy as np
from PIL import Image
import cv2
import time
import mss
from scipy.spatial import distance as dist

"""
okay time for the big hurrah
Implementation TODO:
1. Identify blank Tetris boards (binary)
2. Watch found board and check to see if if makes sense as a board?
3. Find details of board and write to known once done?
"""
error = .1

# give two points, return the slope. Ez Pz
def slope(p1,p2):
    return (p1[1] - p2[1]) / (p1[0] - p2[0])

# checks the slopes from p0 to p1 and p2 to p3 (to check if they are within error of being 0)
# as well as checking diagonal slopes to make sure they are within error of being 2
def tetrisy(c):
    # check diagonal slopes
    one = abs(2 - abs(slope(c[0],c[2])))
    two = abs(2 - abs(slope(c[1],c[3])))

    #check horizontal slopes
    three = abs(slope(c[0],c[1]))
    four = abs(slope(c[2],c[3]))

    if (one <= error) and (two <= error) and (three <= error) and (four <= error):
        return True

# reorders contour points into clockwise order for evaluation
def clock_order(pts):
    # from my good friends at https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def screen_record(timetrial = False, capture='screen', show=False):
    if timetrial:
        last_time = time.time()
    if capture == 'screen':
        sct = mss.mss()
    elif capture == 'cam':
        camera = cv2.VideoCapture(0)

    while(True):
        if capture == 'screen':
            monitor = {"top": 0, "left": 1720, "width": 1720, "height": 1440}
            sct_img = sct.grab(monitor)
            printscreen = np.array(sct_img)
        elif capture == 'cam':
            printscreen = camera.read()[1]
        # convert image to grayscale
        #gray = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
        # smooth the image
        #smooth = cv2.GaussianBlur(gray,(5,5),0)
        # threshold the image - actually a bad idea for color lol

        gray = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)

        #threshhold to normalize some of the weirdness
        #round darks to same
        thresh = cv2.threshold(gray,105,255,cv2.THRESH_TOZERO)[1]
        #round lights to same
        #thresh = cv2.threshold(thresh, 200, 255, cv2.THRESH_TOZERO_INV)[1]

        # edge detection
        edges = cv2.Canny(thresh,100,500)
        # find contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # approximate the polys (if existent)?
        if len(contours) > 0:
            for c in contours:
                # uh
                peri = cv2.arcLength(c, True)
                poly = cv2.approxPolyDP(c,.04 * peri,True)
                if len(poly) == 4 and cv2.contourArea(poly) > 100000:
                    # polygon has 4 vertices, now need to check if it is tetris-y?
                    poly = clock_order(poly.reshape(-1,2))
                    if tetrisy(poly):
                        # at this point, blank board has been found so now need to matrify and learn
                        # approximate true shape/size
                        x1 = (poly[0][0] + poly[3][0])/2
                        y1 = (poly[0][1] + poly[1][1])/2
                        x2 = (poly[3][0] + poly[2][0])/2
                        y2 = (poly[1][1] + poly[2][1])/2
                        # find pixel size of squares
                        px = x2-x1/10
                        py = y2-y1/20
                        # saving these in case I want to draw again
                        #poly = poly.reshape((-1,1,2))
                        #cv2.polylines(printscreen,np.int32([poly]),True,(0,255,0), thickness=3)

        if timetrial:
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()

        # show the screen
        if show:
            cv2.imshow('testing things', printscreen)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


screen_record(show=True)