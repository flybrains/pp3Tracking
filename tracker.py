import numpy as np
import cv2

def track(frame, threshVal, flySize):

	_, thresh = cv2.threshold(frame, threshVal, 255, 0)
	_, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	lastGoodX = None
	lastGoodY = None

	if len(contours)!=0:

		for i, contour in enumerate(contours):

			area = cv2.contourArea(contour)

			if  area > 100 and area <1000:

				M = cv2.moments(contour)

				if M['m00'] != 0:
					cx = int(M['m10']/M['m00'])
					cy = int(M['m01']/M['m00'])

					lastGoodX = cx
					lastGoodY = cy
					success = True

			else:
				if lastGoodX is not None:
					cx = lastGoodX
					cy = lastGoodY
					success = True
				else:
					success = False
	else:
		if lastGoodX is not None:
			cx = lastGoodX
			cy = lastGoodY
			success = False
		else:
			success = False


	if success:
		frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
		frame = cv2.circle(frame, (cx,cy), 3, (0,255,0),2,8,0)
	else:
		frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
		frame = cv2.putText(frame, 'Bad Read', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
		cx = lastGoodX
		cy = lastGoodY


	return success, frame, [cx,cy]
