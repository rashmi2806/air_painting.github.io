import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

######################################
brushThickness=15
eraserThickness=50

#####################################
folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overLayList = []

for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')
    overLayList.append(image)
print(len(overLayList))
header = overLayList[0]
drawColor = (124,252,0)

cap=cv2.VideoCapture(0)
cap.set(1, 640)
cap.set(2, 480)

detector = htm.handDetector(detectionCon= 0.85)
xp,yp=0,0
imgCanvas=np.zeros((480,640,3),np.uint8)

while True:

    #1.Import image
    success, img =cap.read()
    img =cv2.flip(img, 1)

    #2.find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList)!= 0:

       # print(lmList)

        #tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]



        #3.check which fingers are up

        fingers = detector.fingersUp()
        #print(fingers)

        #4.if selection mode -two finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("selection mode")
            if y1 < 62:
                if 110<x1<150:
                    header=overLayList[0]
                    drawColor=(0,255,0)
                elif 200<x1<350:
                    header=overLayList[1]
                    drawColor=(255,0,0)
                elif 400<x1<520:
                    header=overLayList[2]
                    drawColor=(0,0,255)
                elif 530<x1<600:
                    header=overLayList[3]
                    drawColor=(0,0,0)
        cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)



        #5.if dreawing mode-index finger is up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1,y1), 15, drawColor,cv2.FILLED)
            print("drawing mode")
            if xp==0 and yp==0:
                xp,yp=x1,y1

            if drawColor==(0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp,yp=x1,y1

    imgGray=cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv =cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2RGB)
    img=cv2.bitwise_and(img,imgInv)
    img=cv2.bitwise_or(img,imgCanvas)


    #setting the header image
    img[0:62,0:480] = header
    #img=cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image",img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)
