def getPoint(frame):
	f=open(f'ParsedFiles/{frame:05}.txt','r')

	points=[]

	for line in f.readlines():
		nums=line.split(' ')
		cx=(float(nums[3])+float(nums[1]))/2
		cy=(float(nums[4])+float(nums[2]))/2
		point=(int(cx),int(cy))

		if not point in points:
			points.append(point)

	return points

if __name__=="__main__":
	print(getPoint(0))