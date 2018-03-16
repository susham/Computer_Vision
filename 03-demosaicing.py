import numpy
import cv2

# this exercise references "Interactions Between Color Plane Interpolation and Other Image Processing Functions in Electronic Photography" by Adams

numpyInput = cv2.imread(filename='./samples/demosaicing.png', flags=cv2.IMREAD_GRAYSCALE).astype(numpy.float32) / 255.0

print(numpyInput.shape)

numpyOutput = numpy.zeros([numpyInput.shape[0] - 2, numpyInput.shape[1] - 2, 3], numpy.float32)
print(numpyOutput.shape)

# demosaic numpyInput by using bilinear interpolation as shown in the slides and described in section 3.3

# the input has the following beyer pattern, id est that the top left corner is red


# BGBGB ....
# GRGRG ....
# BGBGB ....
# ...........
# ...........




for row in range(1, numpyInput.shape[0] -1):
    for col in range(1, numpyInput.shape[1] -1):

        if (row) % 2 == 1 and (col) % 2 == 1:
            numpyOutput[row-1][col-1][0]=numpyInput[row][col]  #Blue Channel

        if (row%2 == 0 and col % 2 == 1) or (row%2 == 1 and col%2 == 0):
            numpyOutput[row-1][col-1][1] = numpyInput[row][col]    #Green Channel

        if row % 2 == 0 and col % 2 == 0 :
            numpyOutput[row-1][col-1][2] = numpyInput[row][col]            #Red Channel


#calculating the values of empty Blue channel pixels

for i in range(1, numpyOutput.shape[0]-1):
	for j in range(1, numpyOutput.shape[1]-1):
		if i%2 == 0 and j%2 == 1:
			numpyOutput[i][j][2] = (numpyOutput[i-1][j][2] + numpyOutput[i+1][j][2])/2
			numpyOutput[i][j][0] = (numpyOutput[i][j-1][0] + numpyOutput[i][j+1][0])/2
		elif i%2 == 1 and j%2 == 0:
			numpyOutput[i][j][2] = (numpyOutput[i][j-1][2] + numpyOutput[i][j+1][2])/2
			numpyOutput[i][j][0] = (numpyOutput[i-1][j][0] + numpyOutput[i+1][j][0])/2
		elif i%2 == 0 and j%2 == 0:
			numpyOutput[i][j][2] = (numpyOutput[i-1][j-1][2]+ numpyOutput[i+1][j-1][2] + numpyOutput[i-1][j+1][2]+ numpyOutput[i+1][j+1][2])/4
			numpyOutput[i][j][1] = (numpyOutput[i][j-1][1] + numpyOutput[i-1][j][1]+ numpyOutput[i][j+1][1] + numpyOutput[i+1][j][1])/4
		elif i%2 == 1 and j%2 == 1:
			numpyOutput[i][j][0] = (numpyOutput[i-1][j-1][0] + numpyOutput[i+1][j-1][0] + numpyOutput[i-1][j+1][0] + numpyOutput[i+1][j+1][0])/4
			numpyOutput[i][j][1] = (numpyOutput[i][j-1][1] +numpyOutput[i-1][j][1] +numpyOutput[i][j+1][1] + numpyOutput[i+1][j][1])/4


            # the straightforward way that i see for doing this (there are others as well though) is to iterate over each pixel and resolving each of the four possible cases
# to simplify this, you can iterate from (1 to numpyInput.shape[0] - 1) and (1 to numpyInput.shape[1] - 1) to avoid corner cases, numpyOutput is accordingly one pixel smaller on each side

# notice that to fill in the missing greens, you will always be able to take the average of four neighboring values
# however, depending on the case, you either get four or only two neighboring values for red and blue
# this is perfectly fine, in this case you can simply use the average of two values if only two neighbors are available

cv2.imwrite(filename='./03-demosaicing.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))