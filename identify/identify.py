import random
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("identify")
from scipy.spatial import distance
from getPoint import getPoint

class Identify:
	histLen=0
	minDist=0
	hist={}
	frame=0

	def __init__(self, histLen=5, minDist=20):
		self.histLen=histLen
		self.minDist=minDist

	def match(self, points):
		# Objects which have not been seen before
		newPoints={}
		# Objects which have been seen before
		oldPoints={}
		# Objects whose data is stale and must be forgotten
		deadPoints={}

		# Potentially existing points
		if len(self.hist)!=0:
			# Start at prev frame, and iterate to older ones
			for i in range(0,self.histLen):
				if not((self.frame-i) in self.hist):
					continue

				frame = self.hist[self.frame-i]

				removeListIndexes=[]
				for key in frame:
					if len(points)==0:
						break

					# Check for match
					distances=distance.cdist(np.asarray([frame[key]]),np.asarray(points))
					smallestIndex=distances.argmin()
					if ((distances[0][smallestIndex]<self.minDist) and (smallestIndex not in removeListIndexes)):
						oldPoints.update({key: points[smallestIndex]})
						removeListIndexes.append(smallestIndex)

				# Points need to be removed after the looping is done so the indexes don't change
				for index in sorted(removeListIndexes, reverse=True):
					del points[index]

				for key in oldPoints:
					if key in self.hist[self.frame-i]:
						self.hist[self.frame-i].pop(key)

				if len(self.hist[self.frame-i])==0:
					self.hist.pop(self.frame-i)

				self.hist.update({self.frame:oldPoints})

		# Unmatched points are new
		if not points is None:
			for point in points:
				newPoints.update({hex(random.randint(0,2**8)): point})
				self.hist.update({self.frame:newPoints|oldPoints})

		# Points that exist in hist[frame-histLen] are stale
		if (self.frame-self.histLen) in self.hist:
			deadPoints=self.hist[self.frame-self.histLen]

			self.hist.pop(self.frame-self.histLen)

		self.frame+=1
		return (newPoints, oldPoints, deadPoints)

if __name__=="__main__":
	id=Identify(5,20)

	for i in range(0,10):
		# ret=id.match(getPoint(i))
		# print("Frame #",i)
		# print("New:\t",ret[0])
		# print("Old:\t",ret[1])
		# print("Dead:\t",ret[2])
		# print("")

		xarr = [x for x,y in getPoint(i)]
		yarr = [y for x,y in getPoint(i)]

		plt.clf()
		ax=plt.gca()
		ax.set_xlim([0, 500])
		ax.set_ylim([0, 500])
		plt.scatter(xarr,yarr)

		# new, old, dead = ret
		# for pointName in new:
		# 	ax.annotate(pointName, new[pointName])
		# for pointName in old:
		# 	ax.annotate(pointName, old[pointName])

		plt.savefig("img/"+str(i)+".png")
		# plt.show()