
import numpy
import cv2

# this exercise references "Pyramid Methods in Image Processing" by Adelson et al.

numpyFirst = cv2.imread(filename='./samples/multiband-apple.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0
numpySecond = cv2.imread(filename='./samples/multiband-orange.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0

# blend the apple and the orange using multiband blending with paplacian pyramids

# creating a laplacian pyramid with seven levels for each of the two images

numpyFirst = [ numpyFirst ]
numpySecond = [ numpySecond ]

for intLevel in range(6):
	numpyFirst.append(cv2.pyrDown(numpyFirst[-1]))
	numpySecond.append(cv2.pyrDown(numpySecond[-1]))

	numpyFirst[-2] -= cv2.pyrUp(numpyFirst[-1])
	numpySecond[-2] -= cv2.pyrUp(numpySecond[-1])
# end
numpyPyramid = []


for i in range(7):
	numpyPyramid.append(numpy.concatenate((numpyFirst[i][:,: len(numpyFirst[i])/2,:],numpySecond[i][:,len(numpySecond[i])/2:,:]),axis=1))

numpyOutput=numpyPyramid[0]+cv2.pyrUp(numpyPyramid[1]+cv2.pyrUp(numpyPyramid[2]+cv2.pyrUp(numpyPyramid[3]+cv2.pyrUp(numpyPyramid[4]+cv2.pyrUp(numpyPyramid[5]+cv2.pyrUp(numpyPyramid[6]))))))

# combine the two laplacian pyramids and create a new laplacian pyramid to blend the two images
# specifically, take the left half from numpyFirst and the right half from numpySecond at each level
# afterwards, collapse numpyPyramid to obtain the blended result and store it in numpyOutput


cv2.imwrite(filename='./10-multiband.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))
