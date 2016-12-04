import cv2
import numpy as np
from matplotlib import pyplot as plt

fire_cap = cv2.VideoCapture('src/fire.avi')
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = fire_cap.read()

    fgmask = fgbg.apply(frame)
 
    cv2.imshow('fgmask',frame)
    cv2.imshow('frame',fgmask)

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    

fire_cap.release()
cv2.destroyAllWindows()