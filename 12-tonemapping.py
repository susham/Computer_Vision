
import numpy
import cv2
import os
import zipfile
import matplotlib.pyplot

# this exercise references "Photographic Tone Reproduction for Digital Images" by Reinhard et al.

numpyRadiance = cv2.imread(filename='./samples/ahwahnee.hdr', flags=-1)

# perform tone mapping according to the photographic luminance mapping

# first extracting the intensity from the color channels
# note that the eps is to avoid divisions by zero and log of zero

numpyRadianceB=numpyRadiance[:, :,0]
numpyRadianceG=numpyRadiance[:, :,1]
numpyRadianceR=numpyRadiance[:, :,2]


numpyLuminance=numpy.zeros([numpyRadiance.shape[0],numpyRadiance.shape[1]], numpy.float32)


delta=0.001
a=0.18
s=0.6

#Extracting the Luminance values from the color channels.

for intX in range(numpyRadiance.shape[0]):
    for intY in range(numpyRadiance.shape[1]):
        numpyLuminance[intX,intY]= (0.3* numpyRadiance[intX,intY,0] + 0.59 * numpyRadiance[intX, intY,1] + 0.11* numpyRadiance[intX, intY,2])

nLuminanceInput= numpy.copy(numpyLuminance)


numpyIntensity = cv2.cvtColor(src=numpyRadiance, code=cv2.COLOR_BGR2GRAY) + 0.0000001
Lin=numpyIntensity
logedarray=[]
for intX in range(numpyLuminance.shape[0]):
    for intY in range(numpyLuminance.shape[1]):
        logedarray.append(numpy.log(numpyLuminance[intX][intY]+delta))

LW=numpy.exp(numpy.sum(logedarray)/(numpyLuminance.shape[0]* numpyLuminance.shape[1]))


for intX in range(numpyLuminance.shape[0]):
    for intY in range(numpyLuminance.shape[1]):
        numpyLuminance[intX,intY]=(a/LW)*(numpyLuminance[intX,intY])
        numpyLuminance[intX, intY]=(numpyLuminance[intX, intY])/(1+numpyLuminance[intX, intY])



numpyOutput=numpy.zeros([numpyRadiance.shape[0],numpyRadiance.shape[1],3], numpy.float32)
#Color treatment

numpyOutput[:,:,0] = numpy.power((numpyRadianceB/nLuminanceInput),s) * numpyLuminance
numpyOutput[:,:,1] = numpy.power((numpyRadianceG/nLuminanceInput),s) * numpyLuminance
numpyOutput[:,:,2] = numpy.power((numpyRadianceR/nLuminanceInput),s) * numpyLuminance



# start off by approximating the key of numpyIntensity according to equation 1
# then normalize numpyIntensity using a = 0.18 according to equation 2
# afterwards, apply the non-linear tone mapping prescribed by equation 3
# finally obtain numpyOutput using the ad-hoc formula with s = 0.6 from the slides




cv2.imwrite(filename='./12-tonemapping.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))