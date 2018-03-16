import numpy
import cv2

# this exercise references "Seam Carving for Content-Aware Image Resizing" by Avidan and Shamir

numpyInput = cv2.imread(filename='./samples/seam.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0

# implement content-aware image resizing to reduce the width of the image by one-hundred pixels

# using a heuristic energy function to extract an energy map

numpyEnergy = numpy.abs(cv2.Sobel(src=cv2.cvtColor(src=numpyInput, code=cv2.COLOR_BGR2GRAY), ddepth=-1, dx=1, dy=0, ksize=3, scale=1, delta=0.0, borderType=cv2.BORDER_DEFAULT)) \
			+ numpy.abs(cv2.Sobel(src=cv2.cvtColor(src=numpyInput, code=cv2.COLOR_BGR2GRAY), ddepth=-1, dx=0, dy=1, ksize=3, scale=1, delta=0.0, borderType=cv2.BORDER_DEFAULT))




def findMinSeam(CEnergyMap):
	iseam=[]
	mincol = numpy.argmin(CEnergyMap[CEnergyMap.shape[0]-1, :])
	iseam.append(mincol)
	k=mincol
	for row in reversed(range(0,CEnergyMap.shape[0]-1)):
		#Find the minimum value at the top row and use the column value to find the seam.
		if mincol == 0:
			minval = min(CEnergyMap[row - 1, mincol], CEnergyMap[row - 1, mincol + 1])
			if (minval == CEnergyMap[row - 1, mincol + 1]):
					k=mincol + 1
		if mincol == CEnergyMap.shape[1]-1:
			minval=min(CEnergyMap[row - 1, mincol - 1], CEnergyMap[row - 1, mincol])
			if (minval == CEnergyMap[row - 1, mincol - 1]):
				k=mincol - 1
			#iseam.append(mincol)
		if mincol > 0 and mincol < CEnergyMap.shape[1]-1:
			minval = min(CEnergyMap[row - 1, mincol - 1], CEnergyMap[row - 1, mincol], CEnergyMap[row - 1, mincol + 1])
			if(minval== CEnergyMap[row - 1, mincol - 1]):
				k=mincol-1
			if minval == CEnergyMap[row-1, mincol+1]:
				k=mincol+1
		mincol=k
		iseam.append(mincol)
	return iseam


for intRemove in range(100):
	# find and remove one-hundred vertical seams, can potentially be slow
	CEnergyMap = numpy.array(numpyEnergy, copy=True)

	# Update each value of the cEnergyMap, except ignoring corner cases.

	for intX in range(1, CEnergyMap.shape[0]):
		for intY in range(CEnergyMap.shape[1]):
			if intY==0:
				CEnergyMap[intX, intY] = numpyEnergy[intX, intY] + min(CEnergyMap[intX - 1, intY],
																	   CEnergyMap[intX - 1, intY + 1])
			if intY==CEnergyMap.shape[1]-1:
				CEnergyMap[intX, intY] = numpyEnergy[intX, intY] + min(CEnergyMap[intX - 1, intY - 1],
																	   CEnergyMap[intX - 1, intY])
			if intY>0 and intY<CEnergyMap.shape[1]-1:
				CEnergyMap[intX, intY] = numpyEnergy[intX, intY] + min(CEnergyMap[intX - 1, intY - 1],
																   CEnergyMap[intX - 1, intY],
																   CEnergyMap[intX - 1, intY + 1])
	print("Cumulative Energy Map created")



	intSeam = findMinSeam(CEnergyMap)
	intSeam=intSeam[::-1]

	# construct the cumulative energy map using the dynamic programming approach
	# initialize the cumulative energy map by making a copy of the energy map
	# when iterating over the rows, ignore M(y-1, ...) that are out of bounds

	# several seams can have the same energy, use the following for consistency
	# start at the leftmost M(height-1, x) with the lowest cumulative energy
	# should M(y-1, x) be equal to M(y-1, x-1) or M(y-1, x+1) then use (y-1, x)
	# similarly should M(y-1, x-1) be equal to M(y-1, x+1) then use (y-1, x-1)

	# the intSeam array should be a list of integers representing the seam
	# a seam from the top left to the bottom right: intSeam = [0, 1, 2, 3, 4, ...]
	# a seam that is just the first column: intSeam = [0, 0, 0, 0, 0, 0 , ...]





	# some sanity checks, such that the length of the seam is equal to the height of the image
	# furthermore iterating over the seam and making sure that it is a connected sequence

	assert(len(intSeam) == numpyInput.shape[0])

	for intY in range(1, len(intSeam)):
		assert(intSeam[intY] - intSeam[intY - 1] in [-1, 0, 1])
	# end

	# change the following condition to true if you want to visualize the seams that are being removed
	# note that this will not work if you are connected to the linux lab via ssh but no x forwarding

	if False:
		for intY in range(len(intSeam)):
			numpyInput[intY, intSeam[intY], :] = numpy.array([ 0.0, 0.0, 1.0 ], numpy.float32)
		# end

		cv2.imshow(winname='numpyInput', mat=numpyInput)
		cv2.waitKey(10)
	# end

	# removing the identified seam by iterating over each row and shifting them accordingly
	# after the shifting in each row, the image and the energy map are cropped by one pixel on the right

	for intY in range(len(intSeam)):
		numpyInput[intY, intSeam[intY]:-1, :] = numpyInput[intY, (intSeam[intY] + 1 ):, :]
		numpyEnergy[intY, intSeam[intY]:-1] = numpyEnergy[intY, (intSeam[intY] + 1):]
	# end

	numpyInput = numpyInput[:, :-1, :]
	numpyEnergy = numpyEnergy[:, :-1]
# end

cv2.imwrite(filename='./11-seam.png', img=(numpyInput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))