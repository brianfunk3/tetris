import numpy as np
from PIL import Image
import cv2
import time
import mss
from scipy.spatial import distance as dist
import ctypes
import skimage.measure
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from math import sqrt, pow
from itertools import product

recterror = .1
polyerror = .05
eucerror = 65

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

    if (one <= recterror) and (two <= recterror) and (three <= recterror) and (four <= recterror):
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
        user32 = ctypes.windll.user32
        screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    elif capture == 'cam':
        camera = cv2.VideoCapture(0)

    while(True):
        if capture == 'screen':
            monitor = {"top": 0, "left": 0, "width": screensize[0], "height": screensize[1]}
            sct_img = sct.grab(monitor)
            printscreen = np.array(sct_img)
        elif capture == 'cam':
            printscreen = camera.read()[1]
        # convert image to grayscale
        #gray = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
        # smooth the image
        #smooth = cv2.GaussianBlur(gray,(5,5),0)
        # threshold the image - actually a bad idea for color lol
        board = cv2.threshold(printscreen, 130, 255, cv2.THRESH_TOZERO)[1]
        hsvbig = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsvbig)

        #threshhold to normalize some of the weirdness
        #round darks to same
        thresh = cv2.threshold(v,100,255,cv2.THRESH_BINARY)[1]
        #round lights to same
        #thresh = cv2.threshold(thresh, 200, 255, cv2.THRESH_TOZERO_INV)[1]

        # edge detection
        edges = cv2.Canny(thresh,500,800)
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
                        print(poly)
                        # at this point, blank board has been found so now need to matrify and learn
                        # approximate true shape/size
                        x1 = (poly[3][0] + poly[0][0])/2
                        y1 = (poly[3][1] + poly[2][1])/2
                        x2 = (poly[1][0] + poly[2][0])/2
                        y2 = (poly[0][1] + poly[1][1])/2
                        # find pixel size of squares
                        px = (x2-x1)/10
                        py = (y1-y2)/20
                        avp = round((px+py)/2,0)
                        r = 5
                        l = list(product(range(int(x2-r), int(x2+r), 1), range(int(y2-r), int(y2+r), 1)))
                        besterror = px*py*avp*1.5
                        print(besterror)
                        bestpoint = (-1,-1)
                        for pair in l:
                            # cut printscreen to correct dimensions
                            x=pair[0]
                            y=pair[1]
                            cropped = printscreen[y:y + int(20 * avp), x:x + int(10 * avp)]

                            # put it through the blender
                            board = cv2.GaussianBlur(cropped, (5, 5), 5)
                            board = cv2.threshold(board, 130, 255, cv2.THRESH_TOZERO)[1]
                            hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
                            h, s, v = cv2.split(hsv)
                            thresh = cv2.threshold(s, 200, 255, cv2.THRESH_BINARY)[1]

                            #find summation error and compa
                            error = sum(sum(thresh))
                            if error < besterror:
                                besterror = error
                                bestpoint = pair

                        if bestpoint is (-1,-1):
                            print('error is sad, trying again')
                            continue
                        else:
                            print('bestpoint is ' + str(bestpoint) + ' with an error of ' + str(besterror))
                            learn(bestpoint[0],bestpoint[1],avp)

                        # saving these in case I want to draw again
                        #poly = poly.reshape((-1,1,2))
                        #cv2.polylines(printscreen,np.int32([poly]),True,(0,255,0), thickness=3)

        if timetrial:
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()

        # show the screen
        if show:
            cv2.imshow('testing things', edges)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
def threedist(x,y):
    return sqrt(pow((int(x[0])-int(y[0])),2) + pow((int(x[1])-int(y[1])),2) + pow((int(x[2])-int(y[2])),2))

def consolidate(points):
    kmeans = KMeans(n_clusters = 9).fit(points)
    return kmeans.cluster_centers_.tolist()

def learn(x0,y0,p):
    print('found game at ' + str(x0) + ' ' + str(y0) + ' ' + str(p))
    # for trying to learn suspected game
    w = int(10*p)
    h = int(20*p)
    sct = mss.mss()
    monitor = {"top": y0, "left": x0-w, "width": w, "height": int(h)}

    error = eucerror
    perror = polyerror
    success_error = 40
    captures = 0
    colors = [np.array((0,0,0)),np.array((125,125,125))]
    successes = []

    #new method of isolating the pieces to learn color.
    while(True):
        sct_img = sct.grab(monitor)
        board = np.array(sct_img)
        #board = cv2.copyMakeBorder(board, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
        board = cv2.GaussianBlur(board, (5, 5), 5)

        #filter out some grossies
        board = cv2.threshold(board,130,255,cv2.THRESH_TOZERO)[1]
        hsv = cv2.cvtColor(board,cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        thresh = cv2.threshold(s, 200, 255, cv2.THRESH_BINARY)[1]
        #LOOK UP THRESHOLIND THINGS
        captures += 1

        cv2.imshow('testing things', s)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

"""
CODE GRAVEYARD

    # make larger collection array
    collected = None
    captures = 0

    while(captures < 500):
        #print('capping')
        sct_img = sct.grab(monitor)
        board = np.array(sct_img)

        #filter out some grossies
        thresh = cv2.threshold(board, 90, 255, cv2.THRESH_TOZERO)[1]

        #avg pool
        pooledzero = skimage.measure.block_reduce(thresh[:, :, 0], (int(p), int(p)), np.median)
        pooledone = skimage.measure.block_reduce(thresh[:, :, 1], (int(p), int(p)), np.median)
        pooledtwo = skimage.measure.block_reduce(thresh[:, :, 2], (int(p), int(p)), np.median)

        combo = np.dstack((pooledtwo,pooledone,pooledzero))

        if collected is not None:
            collected = np.vstack((collected,combo))
        else:
            collected = combo
"""


screen_record(show=False)
#learn(2299,304,28)