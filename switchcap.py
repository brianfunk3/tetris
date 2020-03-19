# file for getting frames from switch and also probably everything else because yikes

# this is it?
import cv2
import skimage.measure
import numpy as np
import matplotlib.pyplot as plt

# resolution and device
cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)

# mixed allowance
error = .85

# read in saved games
saved = []
with open('saved.txt','r') as f:
    for line in f:
        saved.append([int(i) for i in line.split(sep=':')])
print(saved)

if len(saved) == 0:
    print('no saved gamed smh')

# create successful check list
matches = [0] * len(saved)

# set current game
current = None
currentMisses = 0
sames = 0
prevArea = None
prevGrey = None
tetrises = 0
triples = 0
doubles = 0
singles = 0


while(True):
    ret, frame = cap.read()

    #toss the frame in a blender
    #board = cv2.threshold(frame, 80, 255, cv2.THRESH_TOZERO)[1]
    hsvbig = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsvbig)
    satmask = cv2.threshold(s,90,255,cv2.THRESH_BINARY_INV)[1]
    threshv = cv2.threshold(v,110,255,cv2.THRESH_BINARY)[1] - satmask

    #check error on the frame for each game
    #CLEAN UP LATER!!!!
    if current is None:
        # check for current games
        for i in range(0,len(saved)):
            board = threshv[saved[i][1]:saved[i][1] + int(20 * saved[i][2]), saved[i][0]:saved[i][0] + int(10 * saved[i][2])]
            means = skimage.measure.block_reduce(board, (int(saved[i][2]), int(saved[i][2])), np.mean)/255
            area = np.count_nonzero(means >= error) + np.count_nonzero(means <= (1-error))
            if area >= 195:
                matches[i] = min(matches[i]+1, 15)
            else:
                matches[i] = 0
            if max(matches) >= 15 and sum(matches) < 18:
                print('game found, bet.')
                #find index of max value
                ind = matches.index(max(matches))
                # check that current frame does not equal prev frame for stationary failure
                #if np.count_nonzero(means >= .8) != prevArea:
                    #print('welp')
                current = saved[ind]
    else:
        # check current game errorq
        #cv2.rectangle(frame, (current[0], current[1]), (current[0] + (10 * current[2]), current[1] + (20 * current[2])),
         #             color=(0, 255, 0), thickness=3)
        board = threshv[current[1]:current[1] + int(20 * current[2]), current[0]:current[0] + int(10 * current[2])]
        means = skimage.measure.block_reduce(board, (int(current[2]), int(current[2])), np.mean) / 255
        # check frame consistency
        realarea = np.count_nonzero(means >= error)
        empties = np.count_nonzero(means <= (1-error))
        satboard = satmask[current[1]:current[1] + int(20 * current[2]), current[0]:current[0] + int(10 * current[2])]
        satmeans = skimage.measure.block_reduce(satboard, (int(current[2]), int(current[2])), np.mean) / 255
        greys = np.count_nonzero(satmeans > .7)
        #print('greys: ' + str(greys))
        #print('realarea: ' + str(realarea))

        if prevArea is None:
            if realarea >= 4:
                print('game start')
                prevArea = realarea
                prevGrey = greys
                currentMisses = 0
                continue
            else:
                currentMisses += 1
                continue
        else:
            if realarea < 4:
                currentMisses += 1

        # check for exit
        if currentMisses >= 160:
            print('exiting game :(')

            print('---GAMES STATS---')
            print('TETRISES: ' + str(tetrises))
            print('TRIPLES: ' + str(triples))
            print('DOUBLES: ' + str(doubles))
            print('SINGLES: ' + str(singles))
            
            sames = 0
            currentMisses = 0
            tetrises = 0
            triples = 0
            doubles = 0
            singles = 0
            prevArea = None
            realarea = None
            prevGrey = None
            current = None
            continue

        #print('prevarea is ' + str(prevArea) + ' and current area is ' + str(realarea))

        if realarea+empties >= 190:
            if (prevGrey-greys) > 0 and greys % 9 == 0:
                greyadd = greys/9
            else:
                greyadd = 0
            clears = (prevArea - realarea) + greyadd
            #print(clears)
            if clears >= 40:
                print('TETRIS')
                tetrises += 1
            elif clears >= 30:
                print('TRIPLE')
                triples += 1
            elif clears >= 20:
                print('DOUBLE')
                doubles += 1
            elif clears >= 10:
                print('SINGLE')
                singles += 1
            # if it hits nones of these then its probably some garbage and I don't want it lol

            # update area when sure of everything
            prevArea=realarea
            prevGrey = greys

    #cv2.imshow('Tetris', threshv)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()