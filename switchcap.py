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
error = .1
classerror = .3

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
prevFrame = None
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
    threshv = cv2.threshold(v,110,255,cv2.THRESH_BINARY)[1]

    #check error on the frame for each game
    #CLEAN UP LATER!!!!
    if current is None:
        # check for current games
        for i in range(0,len(saved)):
            board = threshv[saved[i][1]:saved[i][1] + int(20 * saved[i][2]), saved[i][0]:saved[i][0] + int(10 * saved[i][2])]
            means = skimage.measure.block_reduce(board, (int(saved[i][2]), int(saved[i][2])), np.mean)/255
            goods = np.count_nonzero((means < error))
            if goods >= 199:
                matches[i] = min(matches[i]+1, 15)
            else:
                matches[i] = 0
            if max(matches) >= 15 and sum(matches) < 18:
                print('game found, bet.')
                #find index of max value
                ind = matches.index(max(matches))
                # check that current frame does not equal prev frame for stationary failure
                if not np.array_equal(np.around(means,0), prevFrame):
                    #print('welp')
                    current = saved[ind]
    else:
        # check current game errorq
        board = threshv[current[1]:current[1] + int(20 * current[2]), current[0]:current[0] + int(10 * current[2])]
        means = skimage.measure.block_reduce(board, (int(current[2]), int(current[2])), np.mean) / 255
        goods = np.count_nonzero((means < classerror) | (means > (1 - classerror)))
        # check # of error squares
        if goods <= 185:
            currentMisses += 1
        else:
            cv2.rectangle(frame, (current[0], current[1]), (current[0] + (10 * current[2]), current[1] + (20 * current[2])), color=(0, 255, 0), thickness=3)
            currentMisses = 0

        # check frame consistency
        rounded = np.around(means/.5, 0)*.5
        if prevFrame is not None and np.array_equal(rounded, prevFrame):
            #print('sames now at ' + str(sames))
            sames += 1
        else:
            sames = 0

        # check for exit
        if currentMisses >= 30 or sames >= 100:
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
            current = None
        # check for change in top row (IE interpret board)
        if prevFrame is not None and np.sum(prevFrame) != np.sum(rounded):
            clears = (np.count_nonzero(prevFrame == 1) - np.count_nonzero(rounded == 1))
            #print(str(clears))
            if 40 >= clears >= 36:
                print('TETRIS')
                tetrises += 1
            elif 30 >= clears >= 26:
                print('TRIPLE')
                triples += 1
            elif 20 >= clears >= 16:
                print('DOUBLE')
                doubles += 1
            elif 10 >= clears >= 6:
                print('SINGLE')
                singles += 1
            # if it hits nones of these then its probably some garbage and I don't want it lol


        prevFrame = rounded


    #cv2.imshow('Tetris', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()