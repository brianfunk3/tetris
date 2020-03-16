# file for getting frames from switch and also probably everything else because yikes

# this is it?
import cv2
import skimage.measure
import numpy as np

# resolution and device
cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)

# mixed allowance
error = .25

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
            goods = np.count_nonzero((means < error) | (means > (1-error)))
            if goods > 190:
                matches[i] = min(matches[i]+1, 15)
            else:
                matches[i] = 0
            if max(matches) >= 15 and sum(matches) < 18:
                print('game found, bet.')
                #find index of max value
                ind = matches.index(max(matches))
                # check that current frame does not equal prev frame for stationary failure
                if not np.array_equal(np.around(means,0), prevFrame):
                    print('welp')
                    current = saved[ind]
    else:
        # check current game error
        board = threshv[current[1]:current[1] + int(20 * current[2]), current[0]:current[0] + int(10 * current[2])]
        means = skimage.measure.block_reduce(board, (int(current[2]), int(current[2])), np.mean) / 255
        goods = np.count_nonzero((means < error) | (means > (1 - error)))
        # check # of error squares
        if goods <= 185:
            currentMisses += 1
        else:
            cv2.rectangle(frame, (current[0], current[1]), (current[0] + (10 * current[2]), current[1] + (20 * current[2])), color=(0, 255, 0), thickness=3)
            currentMisses = 0

        # check frame consistency
        rounded = np.around(means, 0)
        if prevFrame is not None and np.array_equal(rounded, prevFrame):
            print('sames now at ' + str(sames))
            sames += 1
        else:
            sames = 0
        prevFrame = rounded

        # check for exit
        if currentMisses >= 30 or sames >= 30:
            print('error too high or frames too consistent, back to searching')
            sames = 0
            currentMisses = 0
            current = None


    cv2.imshow('Tetris', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()