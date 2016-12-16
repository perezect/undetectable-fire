import numpy as np
import cv2
import os
import sys

from collections import deque

def convert_video_frame(frame):
    # Converts from video-frame to image-frame
    # input: video frame
    # ret: image frame
	cv2.imwrite("tmp.png", frame)
	tmp = cv2.imread("tmp.png")
	return tmp

def fire_color_thresh(frame, hsv):
	 # Uses an hsv colorspace to threshold for fire-colored objects
     # in: video frame and hsv color space frame
     # ret: image with only fire colored objects 
	sensitivity = 1
	lower_fire = np.array([0, 0, 255-sensitivity], dtype=np.uint8)
	upper_fire = np.array([60, sensitivity, 255], dtype=np.uint8)

	mask = cv2.inRange(hsv, lower_fire, upper_fire)
	res = cv2.bitwise_and(frame, frame, mask= mask)
	return res

def smoke_thresh(frame, hsv):
	# Uses an hsv colorspace to threshold for smoke-colored objects
    # in: video frame and hsv color space frame
    # ret: image with only smoke colored objects 
    # NOTE: this color analysis does not work well, for effective smoke detection
    # we require k-means clustering
	lower_smoke = np.array([200,200,200], dtype=np.uint8)
	upper_smoke = np.array([255,255,255], dtype=np.uint8)
	smask = cv2.inRange(hsv, lower_smoke, upper_smoke)
	res = cv2.bitwise_and(frame, frame, mask= smask)
	return res

def motion_detection(frame, fgbg):
    # Uses mixture of gradients to find foreground mask to detect motion
    # in: video frame and background model
    # ret: foreground mask
	fgmask = fgbg.apply(frame)
	return fgmask

def largest_contours(number, frame):
    # Finds a certian number of contours
    # in: number of top contours, video frame
    # ret: find top contours
	im2, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	num_contours = min(number, len(contours))
	contours = sorted(contours, key = lambda x : x.size, reverse = True)[0:num_contours]
	return contours

def past_frame_queue(num_frames, previous_frames, contours):
    # Fill out a deque of past frames to compare to
    # in: number of past frames, deque, and list of contours
	if len(previous_frames) == num_frames:
		previous_frames.popleft()
		previous_frames.append(contours)
	else:
		previous_frames.append(contours)

def contour_intersection(threat_lvl, capSize, previous_frames, frame, area):
    # Find overlapping contours and determined threat level and conditions of room
    # Writes conditions and threats onto video frame
    # in: amount of overlap threshold, size of capture, deque, video frame, and min area of contour
	# Create an image filled with zeros, single-channel, same size as img.
	text = "SAFE"
	blank = np.zeros((capSize[1], capSize[0]))
	intersection = blank.copy()

	for lst in previous_frames:
		for index in range(len(lst)):
			img = cv2.drawContours(blank.copy(), lst, index, 1, -1)
			intersection = intersection + img

	overlap = np.amax(intersection)

	if overlap > threat_lvl:
		text = "DANGER"
		thresh = np.array(np.where(intersection >= threat_lvl, 1, 0))
		im_thresh = np.array(thresh * 255, dtype = np.uint8)
		im_thresh = im_thresh.copy()
	# print thresh.shape
	# print fgmask.shape

		im3,  cnts, heir = cv2.findContours(im_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

		for c in cnts:
	# if the contour is too small, ignore it
			if cv2.contourArea(c) < area:
				continue

	# compute the bounding box for the contour, draw it on the frame,
	# and update the text
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


	threat = min(overlap/3, 10.0)

	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	cv2.putText(frame, "Threat Level: {}".format(threat), (10, 40),
	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def main():
	####################
	## INITIALIZATION ##
	####################
	cv2.ocl.setUseOpenCL(False)
	# Background model using mixture of gradient models
	# Shadow detection set to false as we are not concerned with shadows
	fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
	if len(sys.argv) == 2:
		cap = cv2.VideoCapture(str(sys.argv[1]))
		# Output to file
		fps = int(cap.get(cv2.CAP_PROP_FPS))
		capSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
		out = cv2.VideoWriter()

		success = out.open('output.mov',fourcc,fps,capSize,True)
		out_bw = cv2.VideoWriter()
		success_2 = out_bw.open('output_bw.mov',fourcc,fps,capSize,True)

		num_frames = 30
		previous_frames = deque()
		MIN_AREA = 500

		# Fire detection algorithm
		while(1):
			####################
			## Color analysis ##
			####################

			# Change colorspace from RGB to HSV (hue saturation value)
			try:
				ret, frame = cap.read()
				hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			except:
				break
			# Thresholding fire-color object
			fire_extract_image = fire_color_thresh(frame, hsv)

			# DEMO PURPOSES
			# cv2.imshow('frame', frame)
			# cv2.imshow('color_mask', res)
			# cv2.imshow('fgmask', fgmask)

			############################
			## Background subtraction ##
			############################

			fgmask = motion_detection(fire_extract_image, fgbg)

			if out is not None:
				tmp = convert_video_frame(frame)
				out_bw.write(tmp)

			#######################
			## Contour detection ##
			#######################

			# Decided on selecting top 3 contours based off of trial and error
			num_contours = 3
			contours = largest_contours(num_contours, fgmask)

			# Draw largest contours
			for i in range(len(contours)):
				cv2.drawContours(frame, contours, i, (0,0,255), 3)

			past_frame_queue(num_frames, previous_frames, contours)
			threat_lvl = 20
			overlap = contour_intersection(threat_lvl, capSize, previous_frames, frame, MIN_AREA)

			cv2.imshow('Is there a fire?', frame)

			if out is not None:
				convert_video_frame(frame)
				out.write(frame)

			k = cv2.waitKey(30) & 0xff
			if k == 27:
				break
		os.remove("tmp.png")
		cap.release()

		if out is not None:
			out.release()
			out_bw.release()

		cv2.destroyAllWindows()
	else:
		print "Too many arguments"


if __name__ == "__main__": main()
