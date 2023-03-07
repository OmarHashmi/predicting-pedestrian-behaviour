import cv2

from identify.identify import Identify
from kalman.kalman import KalmanFilter
from yolo.yolo import Yolo

if __name__ == '__main__':
	# Initialization and Variable Defs
	yoloModel = Yolo("yolo/yolov7.pt")
	identifyModel = Identify(5,20)
	kalmanModel = KalmanFilter()

	# Video Capture Loop
	cap = cv2.VideoCapture("Assets/mot.mp4")
	frameNum=-1
	while (True):
		frameNum+=1
		print(f"Frame: {frameNum}")

		ret, frame=cap.read()
		if (ret==False):
			break

		# Get Object Coords from Yolo
		imgArray=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		coordsList=yoloModel.detectFrame(imgArray)

		# Match Objects Using Identify
		new,old,dead = identifyModel.match(coordsList)
		print(f"New:  {new}\nOld:  {old}\nDead: {dead}")

		# Track Positions Using Kalman
		prediction = kalmanModel.processObjs(new,old,dead)

		# Output Positions to Central Unit
		RED=(0, 0, 255)
		BLUE=(255, 0, 0)
		for p in new:
			cv2.circle(frame, new[p], 3, RED, -1)
		for p in old:
			cv2.circle(frame, old[p], 3, RED, -1)
		for p in prediction:
			cv2.circle(frame, prediction[p], 3, BLUE, -1)

		cv2.imshow("Output", frame)

		if cv2.waitKey(1)&0xFF==ord('q'):
			break
