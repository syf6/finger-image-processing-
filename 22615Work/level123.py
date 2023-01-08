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
    upper_hsv_1 = np.array([20,255,255]) #human skin color mask
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
    print('Grab\n')
   #cv2.putText(img, "grab", (0, img.shape[0] - 10), font, 2, (0, 0, 255), 2)


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

            char = cv2.waitKey(10)
            if char == 27:
                break
        cv2.destroyAllWindows()
        cap.release()


