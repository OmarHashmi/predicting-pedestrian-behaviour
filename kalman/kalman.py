import cv2
import numpy as np

RED = (0,0,255)
BLUE = (255,0,0)

# def vectorSkipped():
# 	cap=cv2.VideoCapture(fName)
#
# 	# setupOptions()
# 	kf=KalmanFilter()
#
# 	frameCounter=0
# 	frameLimit=12
# 	while (True):
# 		ret, frame=cap.read()
# 		if (ret==False):
# 			break
#
# 		if(frameCounter<frameLimit):
# 			frameCounter+=1
# 			continue
# 		else:
# 			frameCounter=0
#
# 		x1, y1, x2, y2=tr.detect(frame)
# 		center=(int((x1+x2)/2), int((y1+y2)/2))
#
# 		cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
# 		cv2.circle(frame, center, 3, RED, -1)
#
# 		prediction=kf.predict(center[0], center[1])
#
# 		dx=prediction[0]-center[0]
# 		dy=prediction[1]-center[1]
#
# 		cv2.circle(frame, (center[0]+(24*dx),center[1]+(4*dy)), 3, BLUE, -1)
#
# 		cv2.imshow("Demo", frame)
#
# 		if cv2.waitKey(1)&0xFF==ord('q'):
# 			break

class KalmanFilter:
	objects={}

	def processObjs(self,new,old,dead):
		predictions ={}

		for objKey in new:
			kf=cv2.KalmanFilter(4, 2)
			kf.measurementMatrix=np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
			kf.transitionMatrix=np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

			self.objects.update({objKey:kf})
			#predictions.update(new[objKey])

		for objKey in old:
			prediction = self.predict(old[objKey][0], old[objKey][1], self.objects[objKey])
			predictions.update({objKey:prediction})

		for objKey in dead:
			self.objects.pop(objKey)

		return predictions

	def predict(self, coordX, coordY, kf):
		measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
		kf.correct(measured)
		predicted = kf.predict()
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