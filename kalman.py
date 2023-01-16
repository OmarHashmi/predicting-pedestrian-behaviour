import cv2
import numpy as np

fName       = "Assets/trackedL.mp4"
frameWidth  = 960
frameHeight = 720

RED = (0,0,255)
BLUE = (255,0,0)

def main():
	normal()
	chained()
	vectored()
	skipped()
	vectorSkipped()

def normal():
	cap=cv2.VideoCapture(fName)

	# setupOptions()
	kf=KalmanFilter()
	tr=Tracker()

	while (True):
		ret, frame=cap.read()
		if (ret==False):
			break

		x1, y1, x2, y2=tr.detect(frame)
		center=(int((x1+x2)/2), int((y1+y2)/2))

		cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
		cv2.circle(frame, center, 3, RED, -1)

		prediction=kf.predict(center[0], center[1])
		cv2.circle(frame, prediction, 3, BLUE, -1)

		cv2.imshow("Demo", frame)

		if cv2.waitKey(1)&0xFF==ord('q'):
			break

def chained():
	cap=cv2.VideoCapture(fName)

	# setupOptions()
	kf=KalmanFilter()
	tr=Tracker()

	while (True):
		ret, frame=cap.read()
		if (ret==False):
			break

		x1, y1, x2, y2=tr.detect(frame)
		center=(int((x1+x2)/2), int((y1+y2)/2))

		cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
		cv2.circle(frame, center, 3, RED, -1)

		prediction=kf.predict(center[0], center[1])
		p2=kf.predict(prediction[0], prediction[1])
		# for i in range(3):
		# 	p2=kf.predict(p2[0],p2[1])
		# 	print(p2)

		cv2.circle(frame, p2, 3, BLUE, -1)

		cv2.imshow("Demo", frame)

		if cv2.waitKey(1)&0xFF==ord('q'):
			break

def vectored():
	cap=cv2.VideoCapture(fName)

	# setupOptions()
	kf=KalmanFilter()
	tr=Tracker()

	while (True):
		ret, frame=cap.read()
		if (ret==False):
			break

		x1, y1, x2, y2=tr.detect(frame)
		center=(int((x1+x2)/2), int((y1+y2)/2))

		cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
		cv2.circle(frame, center, 3, RED, -1)

		prediction=kf.predict(center[0], center[1])

		dx=prediction[0]-center[0]
		dy=prediction[1]-center[1]

		cv2.circle(frame, (center[0]+(24*dx),center[1]+(24*dy)), 3, BLUE, -1)

		cv2.imshow("Demo", frame)

		if cv2.waitKey(1)&0xFF==ord('q'):
			break

def skipped():
	cap=cv2.VideoCapture(fName)

	# setupOptions()
	kf=KalmanFilter()
	tr=Tracker()

	frameCounter=0
	frameLimit=12
	while (True):
		ret, frame=cap.read()
		if (ret==False):
			break

		if(frameCounter<frameLimit):
			frameCounter+=1
			continue
		else:
			frameCounter=0

		x1, y1, x2, y2=tr.detect(frame)
		center=(int((x1+x2)/2), int((y1+y2)/2))

		cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
		cv2.circle(frame, center, 3, RED, -1)

		prediction=kf.predict(center[0], center[1])

		cv2.circle(frame, prediction, 3, BLUE, -1)

		cv2.imshow("Demo", frame)

		if cv2.waitKey(1)&0xFF==ord('q'):
			break

def vectorSkipped():
	cap=cv2.VideoCapture(fName)

	# setupOptions()
	kf=KalmanFilter()
	tr=Tracker()

	frameCounter=0
	frameLimit=12
	while (True):
		ret, frame=cap.read()
		if (ret==False):
			break

		if(frameCounter<frameLimit):
			frameCounter+=1
			continue
		else:
			frameCounter=0

		x1, y1, x2, y2=tr.detect(frame)
		center=(int((x1+x2)/2), int((y1+y2)/2))

		cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
		cv2.circle(frame, center, 3, RED, -1)

		prediction=kf.predict(center[0], center[1])

		dx=prediction[0]-center[0]
		dy=prediction[1]-center[1]

		cv2.circle(frame, (center[0]+(24*dx),center[1]+(4*dy)), 3, BLUE, -1)

		cv2.imshow("Demo", frame)

		if cv2.waitKey(1)&0xFF==ord('q'):
			break

class Tracker:
	def __init__(self):
		self.low = np.array([0, 0, 0])
		self.high = np.array([0, 0, 0])

	def detect(self, frame):
		# minH = cv2.getTrackbarPos("Min H", "Params")
		# minS = cv2.getTrackbarPos("Min S", "Params")
		# minV = cv2.getTrackbarPos("Min V", "Params")
		# maxH = cv2.getTrackbarPos("Max H", "Params")
		# maxS = cv2.getTrackbarPos("Max S", "Params")
		# maxV = cv2.getTrackbarPos("Max V", "Params")
		minH=51
		minS=204
		minV=204
		maxH=102
		maxS=255
		maxV=255

		self.low = np.array([minH, minS, minV])
		self.high = np.array([maxH, maxS, maxV])

		hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		mask = cv2.inRange(hsv_img, self.low, self.high)

		contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

		box = (0, 0, 0, 0)
		for cnt in contours:
			(x, y, w, h) = cv2.boundingRect(cnt)
			box = (x, y, x + w, y + h)
			break

		return box

class KalmanFilter:
	kf = cv2.KalmanFilter(4, 2)
	kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
	kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

	def predict(self, coordX, coordY):
		measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
		self.kf.correct(measured)
		predicted = self.kf.predict()
		x, y = int(predicted[0]), int(predicted[1])
		return x, y

def setupOptions():
	cv2.namedWindow("Params")
	cv2.resizeWindow("Params",480,480)
	# cv2.createTrackbar("Min H","Params",0,255,lambda x:None)
	# cv2.createTrackbar("Max H","Params",0,255,lambda x:None)
	# cv2.createTrackbar("Min S","Params",0,255,lambda x:None)
	# cv2.createTrackbar("Max S","Params",0,255,lambda x:None)
	# cv2.createTrackbar("Min V","Params",0,255,lambda x:None)
	# cv2.createTrackbar("Max V","Params",0,255,lambda x:None)

if __name__=="__main__":
	main()