import sys
import numpy as np
import cv2
import math
import time
import random
import os

font=cv2.FONT_HERSHEY_SIMPLEX
#calculate the contour area
def cnt_area(cnt):
  area = cv2.contourArea(cnt)
  return area

def Gesture_Recognize(img):
    humanResult = ""
    pointNum = 1
    hsv_img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_hsv_1 = np.array([0,50,50])
    upper_hsv_1 = np.array([20,255,255])#skin color mask
    lower_hsv_2 = np.array([150,50,50])
    upper_hsv_2 = np.array([180,255,255])
    mask1 = cv2.inRange(hsv_img,lower_hsv_1,upper_hsv_1)
    mask2 = cv2.inRange(hsv_img,lower_hsv_2,upper_hsv_2)
    mask = mask1 + mask2 
    mask = cv2.medianBlur(mask,5)
    k1=np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, k1, iterations=1)
    mask = cv2.erode(mask, k1, iterations=1)
    mask_color = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    cv2.imshow('mask', mask)
    #cv2.imwrite('mask.png', mask)
    black_img = np.zeros(mask.shape,np.uint8)

    contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 1:
        return 0, img
    contours = list(contours) #bug  
    contours.sort(key = cnt_area, reverse=True)  
    (x0, y0, w0, h0) = cv2.boundingRect(contours[0])
    if(w0>=100 and h0>=100):
        cv2.rectangle(img,(x0,y0),(x0+w0,y0+h0),(255,0,255),2)
        epsilon = 0.01 * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0],epsilon,True)
        cv2.drawContours(black_img,[approx],-1,(255,0,255),2)

        contours2,hierarchy2 = cv2.findContours(black_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 1:
          return 0, img
        hull = cv2.convexHull(contours2[0],returnPoints=False)
        defects = cv2.convexityDefects(contours2[0],hull)

        if defects is None:
          #print ('have no convex defects')
          pass
        else: 
          #print ('have convex defects')
          for i in range(0,defects.shape[0]):
            s,e,f,d = defects[i,0]
            pre_start = (0,0)
            pre_end = (0,0)
            start = tuple(contours2[0][s][0])
            end = tuple(contours2[0][e][0])
            far = tuple(contours2[0][f][0])
            #print(d)
            if d >= 13000:
              cv2.line(img,start,end,[0,255,0],3)#convex hull
              cv2.circle(img,start,10,[0,255,255],3)
              cv2.circle(img,end,10,[0,255,255],3)
              cv2.circle(img,far,10,[0,0,255],3)#convex hull defect point
              pre_start = start
              pre_end = end
              pointNum += 1

    #0 or 1 finger: rock, 2 fingers sciscors, 5 fingers: paper
    if pointNum <= 1:
      cv2.putText(img, "rock", (10, img.shape[0]-10), font, 1, (255,255,255), 2)
      humanResult = "rock"
    elif pointNum == 2:
      cv2.putText(img, "scissor", (10, img.shape[0]-10), font, 1, (255,255,255), 2)
      humanResult = "scissor"
    elif pointNum == 5:
      cv2.putText(img, "paper", (10, img.shape[0]-10), font, 1, (255,255,255), 2)
      humanResult = "paper"
    else:
      cv2.putText(img, "bad", (0, img.shape[0]-10), font, 1, (255, 255, 255), 2)

    
     
    cv2.putText(img,'hand-%d'%pointNum,(10,35),font,1.2,(0,255,255),3)       
    return pointNum, img

def move(pointNum):
  #hand movement 1: finger number from 1 to 5 is 'grab'
  record = []
  record.append(pointNum)
  if record[0] == 1 and record[-1] == 5 and record[1] == 1:
   cv2.putText(img, "grab", (0, img.shape[0] - 10), font, 2, (0, 0, 255), 2)


def game():
    choice = []
    choice.append("rock")
    choice.append("scissor")
    choice.append("paper")
    
    capture = cv2.VideoCapture(0)
    
    cv2.imshow('camera', 1)
    start_time = time.time()
    print("----------put your hand in the rectangle in 8 seconds---------\n")
    print("---------------------------------------------\n")
    while (1):
        ha, img = capture.read()
        end_time = time.time()
        cv2.rectangle(img, (426, 0), (640, 250), (170, 170, 0))
        cv2.putText(img, str(int((8 - (end_time - start_time)))), (100, 100), font, 2, 255)
        cv2.imshow('camera', img)
        if (end_time - start_time > 8):
            break
        if (cv2.waitKey(30)>=0):
            break
    ha,img=capture.read()
    capture.release()
    cv2.imshow('camera', img)
    
    img1 = img[0:210, 426:640]
    cv2.imwrite("a1.jpg", img1)
    p1 = Gesture_Recognize()
    pc = random.randint(1, 2, 5)
    print("your choice is", choice[p1], " the computer choice is ", choice[pc], "\n")
    
    if (p1 == pc):
        print("Tie\n")
        return 0
    if ((p1 == 1 and pc == 2) or (p1 == 2 and pc == 5) or (p1 == 5 and pc == 1)):
        print('you win\n')
        return 1
    else:
        print('you lose\n')
        return -1

if __name__ == '__main__':
    cap = cv2.VideoCapture()
    flag = cap.open(0)
    if flag:
        while True:
            ret, frame = cap.read()
            if frame is None:
                break
            cv2.imshow('frame',frame)
            num, img = Gesture_Recognize(frame)
            move(num)
            cv2.imshow('Gesture_Recognize',img)
            
            you_win=0
            pc_win=0
            flag=1
            print("*********this is a game that use camera to play rock paper sciscor with pc**********\n")
            while(flag):
              print("---------------------------------------------\n")
              print("--------------press enter to start the game---------------\n")
              print("--to reduce misjudge, put your hand in the rectangle as much as possible --\n")
              #os.system('cls')
              ans =game()
              
              key=cv2.waitKey(5000)
              if(key==13):#press enter to continue
                  flag=1
              elif(key==27):#press ESC to quit
                flag=0
              elif(key==113):#press q to pause
                cv2.waitKey(0)
              cv2.destroyAllWindows()
              if (ans == 1):
                you_win += 1
              elif (ans == -1):
                pc_win += 1
              if(cv2.waitKey()==27):
                break
              print("***********score(you : PC)", you_win, ":", pc_win, "*******************"'\n')
              print("Game Over")

            char = cv2.waitKey(10)
            if char == 27:
                break
        cv2.destroyAllWindows()
        cap.release()


