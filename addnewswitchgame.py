import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)

print("press q when board is in sight ;)")

while(True):
    ret, frame = cap.read()

    cv2.imshow('Tetris',frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

#frame = cv2.imread("fulllook.jpg")

refPt = []
cropping = False

print()
print("click in the upper left corner of the full board (no half squares or anything I stg)")
print("press r to reset the dot boys")
print("press y to confirm")

clone = frame.copy()

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(frame, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("Lets get it you bitch", frame)

# load the image, clone it, and setup the mouse callback function
cv2.namedWindow("Lets get it you bitch")
cv2.setMouseCallback("Lets get it you bitch", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    if refPt != []:
        cv2.circle(frame, refPt[0], 2, (255, 255, 255))
        cv2.imshow("Lets get it you bitch", frame)
    else:
        cv2.imshow("Lets get it you bitch", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        refPt = []
        frame = clone.copy()

    # if the 'c' key is pressed, break from the loop
    elif key == ord("y") and refPt != []:
        print("YEETTT")

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

# if there are two reference points, then crop the region of interest
# from the image and display it
if len(refPt) == 2:
    width = refPt[1][0] - refPt[0][0]
    height = refPt[1][1] - refPt[0][1]
    wsqr = int(round(width/10))
    hsqr = int(round(height/20))

    print(wsqr)
    print(hsqr)

    if wsqr != hsqr:
        print("square sizes do not match please restart cause Brian is a lazy trash bag :)")
        exit()


#cap.release()
cv2.destroyAllWindows()

cleaned = clone[refPt[0][1]:(refPt[0][1] + (20*wsqr)), refPt[0][0]:(refPt[0][0]+(10*wsqr))]
togglegrid = False

while(True):
    if togglegrid:
        roi = clone[refPt[0][1]:(refPt[0][1] + (20 * wsqr)), refPt[0][0]:(refPt[0][0] + (10 * wsqr))].copy()
        for i in range(1,10):
            cv2.line(roi,(i*wsqr,0),(i*wsqr,700),(255,0,0),1)
        for i in range(1,20):
            cv2.line(roi, (0,i*wsqr), (700,i*wsqr), (255,0,0), 1)
        cv2.imshow("ROI", roi)
    else:
        roi = clone[refPt[0][1]:(refPt[0][1] + (20 * wsqr)), refPt[0][0]:(refPt[0][0] + (10 * wsqr))].copy()
        cv2.imshow("ROI", roi)
    key = cv2.waitKey(1) & 0xFF

    # nah
    if key == ord("w"):
        refPt = ((refPt[0][0], refPt[0][1] - 1), (refPt[1][0], refPt[1][1]))

    # nah
    elif key == ord("a") and refPt != []:
        refPt = ((refPt[0][0] - 1, refPt[0][1]), (refPt[1][0], refPt[1][1]))

    # nah
    elif key == ord("s"):
        refPt = ((refPt[0][0], refPt[0][1] + 1), (refPt[1][0], refPt[1][1]))

    # nah
    elif key == ord("d"):
        refPt = ((refPt[0][0] + 1, refPt[0][1]), (refPt[1][0], refPt[1][1]))

    # toggle a grid
    elif key == ord("t"):
        togglegrid = not togglegrid

    # accept and send
    elif key == ord("c"):
        break

cv2.destroyAllWindows()

print('upper left point is ' + str(refPt[0]) + ' and the point size is ' + str(wsqr))

with open('saved.txt', 'a') as the_file:
    the_file.write(str(refPt[0][0]) + ':' + str(refPt[0][1]) + ':' + str(wsqr))
    the_file.write("\n")

#Proof of concept print
#resized_image = cv2.resize(tester, (400, 800))
#cv2.imwrite("Uhhhh.jpg",resized_image)