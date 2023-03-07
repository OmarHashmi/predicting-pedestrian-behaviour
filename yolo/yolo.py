import argparse
import time
import cv2
import numpy as np
import torch
import sys

sys.path.append("yolo")
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier,scale_coords, set_logging
from utils.torch_utils import select_device, load_classifier, TracedModel


class Yolo:
	def __init__(self,weigths):
		parser=argparse.ArgumentParser()
		parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
		parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
		parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
		parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
		parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
		parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
		parser.add_argument('--view-img', action='store_true', help='display results')
		parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
		parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
		parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
		parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
		parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
		parser.add_argument('--augment', action='store_true', help='augmented inference')
		parser.add_argument('--update', action='store_true', help='update all models')
		parser.add_argument('--project', default='runs/detect', help='save results to project/name')
		parser.add_argument('--name', default='exp', help='save results to project/name')
		parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
		parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

		self.args = parser.parse_args([
			'--weights',weigths,
			'--conf','0.3'
		])

		# Initialize
		set_logging()
		self.device=select_device(self.args.device)
		self.half=self.device.type!='cpu'  # half precision only supported on CUDA

		# Load model
		self.model=attempt_load(self.args.weights, map_location=self.device)  # load FP32 model
		self.stride=int(self.model.stride.max())  # model stride
		self.imgsz=check_img_size(self.args.img_size, s=self.stride)  # check img_size

		if not self.args.no_trace:
			self.model=TracedModel(self.model, self.device, self.args.img_size)

		if self.half:
			self.model.half()  # to FP16

		# Second-stage classifier
		self.classify=False
		if self.classify:
			self.modelc=load_classifier(name='resnet101', n=2)  # initialize
			self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

		# Get names
		self.names=self.model.module.names if hasattr(self.model, 'module') else self.model.names

	def detectFrame(self, frame):
		# Run inference
		if self.device.type!='cpu':
			model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(model.parameters())))  # run once
		old_img_w=old_img_h=self.imgsz
		old_img_b=1

		startTime=0
		t0=time.time()

		# Old loop used to be here
		# im0s=cv2.imread(frame)
		im0s=frame
		img=np.transpose(letterbox(im0s, self.args.img_size, self.stride)[0], (2,0,1))

		img=torch.from_numpy(img).to(self.device)
		img=img.half() if self.half else img.float()  # uint8 to fp16/32
		img/=255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension()==3:
			img=img.unsqueeze(0)

		# Warmup
		if self.device.type!='cpu' and (old_img_b!=img.shape[0] or old_img_h!=img.shape[2] or old_img_w!=img.shape[3]):
			old_img_b=img.shape[0]
			old_img_h=img.shape[2]
			old_img_w=img.shape[3]
			for i in range(3):
				self.model(img, augment=self.args.augment)[0]

		# Inference
		with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
			pred=self.model(img, augment=self.args.augment)[0]

		# Apply NMS
		pred=non_max_suppression(pred, self.args.conf_thres, self.args.iou_thres, classes=self.args.classes,
								 agnostic=self.args.agnostic_nms)

		# Apply Classifier
		if self.classify:
			pred=apply_classifier(pred, self.modelc, img, im0s)

		# Process detections
		ret=[]
		for i, det in enumerate(pred):  # detections per image
			s, im0='', im0s

			if len(det):
				# Rescale boxes from img_size to im0 size
				det[:, :4]=scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

				# Print results
				for c in det[:, -1].unique():
					n=(det[:, -1]==c).sum()  # detections per class
					s+=f"{n} {self.names[int(c)]}{'s'*(n>1)}, "  # add to string

				# Write results
				for *xyxy, conf, cls in reversed(det):
					label=f'{self.names[int(cls)]} {conf:.2f}'
					if "person" in label:
						# print(f"xyxy: {xyxy}")
						# Convert xyxy to cxcy
						cx=(xyxy[0]+xyxy[2])/2
						cy=(xyxy[1]+xyxy[3])/2
						ret.append((int(cx.item()),int(cy.item())))

		return ret

if __name__ == '__main__':
	model = Yolo("yolov7.pt")

	cap = cv2.VideoCapture("../Assets/iphone8.mp4")
	count=-1
	while (True):
		count+=1

		ret, frame=cap.read()
		if (ret==False):
			break

		imgArray=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		ret=model.detectFrame(imgArray)
		print(f"Frame {count}")
		if len(ret)!=0:
			objNum=0
			for coords in ret:
				objNum+=1
				print(f"\tObj #{objNum}:(x1,y1,x2,y2): {coords[0]},{coords[1]},{coords[2]},{coords[3]}")
		else:
			print("\tNo Objects Detected")