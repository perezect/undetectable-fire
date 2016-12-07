import numpy as np
import cv2

cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture('src/roman_candle.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
# try:
fps = int(cap.get(cv2.CAP_PROP_FPS))/20
capSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
out = cv2.VideoWriter()
success = out.open('output.mov',fourcc,fps,capSize,True)
# except:
#     out = None
#     pass

while(1):
    ret, frame = cap.read()

    # Change colorspace from RGB to HSV (hue saturation value)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold for fire
    sensitivity = 100
    lower_fire = np.array([0, 0, 255-sensitivity], dtype=np.uint8)
    upper_fire = np.array([60, sensitivity, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_fire, upper_fire)
    res = cv2.bitwise_and(frame, frame, mask= mask)

    # Threshold for smoke
    lower_smoke = np.array([200,200,200], dtype=np.uint8)
    upper_smoke = np.array([255,255,255], dtype=np.uint8)

    smask = cv2.inRange(hsv, lower_smoke, upper_smoke)

    # Background subtraction
    fgmask = fgbg.apply(res)


    cv2.imshow('fgmask', fgmask)

    im2, contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    num_contours = min(3, len(contours))

    contours = sorted(contours, key = lambda x : x.size, reverse = True)

    for i in range(num_contours):
        cv2.drawContours(frame, contours, i, (0,0,255), 3)

    cv2.imshow('contours', frame)
    if out is not None:
        out.write(frame)

    # Threshold for smoke
    # Likely needs tinkering of the hue values
    # lower_smoke = np.array([200,200,200], dtype=np.uint8)
    # upper_smoke = np.array([255,255,255], dtype=np.uint8)

    # smask = cv2.inRange(hsv, lower_smoke, upper_smoke)
    # smres = cv2.bitwise_and(frame, frame, mask= smask)

    # smmask = fgbg.apply(smres)

    # cv2.imshow('smmask', smmask)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()