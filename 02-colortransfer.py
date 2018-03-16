import numpy
import cv2
import math
# this exercise references "Color Transfer between Images" by Reinhard et al.

numpyFrom = cv2.imread(filename='./samples/transfer-from.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0
numpyTo = cv2.imread(filename='./samples/transfer-to.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0

numpyFrom = cv2.cvtColor(src=numpyFrom, code=cv2.COLOR_BGR2Lab)
numpyTo = cv2.cvtColor(src=numpyTo, code=cv2.COLOR_BGR2Lab)


# l,a,b channels of numpyfrom(transfer-from.png) image.
nf_l= numpy.array(numpyFrom[:, :, 0])
nf_Alpha = numpy.array(numpyFrom[:, :, 1])
nf_Beta = numpy.array(numpyFrom[:, :, 2])




# l,a,b channels  of numpyTo(transfer-to.png) image.
nt_l = numpy.array(numpyTo[:, :, 0])
nt_Alpha = numpy.array(numpyTo[:, :, 1])
nt_Beta = numpy.array(numpyTo[:, :, 2])


# Calculating mean of each LAlphaBeta channel of numpyTo
nt_lmean=numpy.mean(nt_l)
nt_Alphamean=numpy.mean(nt_Alpha)
nt_Betamean=numpy.mean(nt_Beta)

# Calculating mean of each LAlphaBeta channel of numpyFrom
nf_lmean = numpy.mean(nf_l)
nf_Alphamean = numpy.mean(nf_Alpha)
nf_Betamean = numpy.mean(nf_Beta)

# subtract mean from the numpyTo data points
snt_l=nt_l-nt_lmean
snt_Alpha=nt_Alpha-nt_Alphamean
snt_Beta=nt_Beta-nt_Betamean

# calculating the standard deviation of both NumpyTo and NumpyFrom
ntsd_l=numpy.std(nt_l)
ntsd_Alpha=numpy.std(nt_Alpha)
ntsd_Beta=numpy.std(nt_Beta)


nfsd_l=numpy.std(nf_l)
nfsd_Alpha=numpy.std(nf_Alpha)
nfsd_Beta=numpy.std(nf_Beta)


# Scaling numpyTo
#sl = (ntsd_l/nfsd_l)*snt_l
#sAlpha = (ntsd_Alpha/nfsd_Alpha)*snt_Alpha
#sBeta = (ntsd_Beta/nfsd_Beta)*snt_Beta

# Scaling numpyTo
sl = (nfsd_l/ntsd_l)*snt_l
sAlpha = (nfsd_Alpha/ntsd_Alpha)*snt_Alpha
sBeta = (nfsd_Beta/ntsd_Beta)*snt_Beta

r_l=sl+nf_lmean
r_Alpha=sAlpha+nf_Alphamean
r_Beta=sBeta+nf_Betamean


numpyTo=numpy.stack((r_l, r_Alpha, r_Beta), -1)



# match the color statistics of numpyTo to those of numpyFrom


# calculate the per-channel mean of the data points / pixels of numpyTo, and subtract these from numpyTo according to equation 10



# calculate the per-channel std of the data points / pixels of numpyTo and numpyFrom, and scale numpyTo according to equation 11
# calculate the per-channel mean of the data points / pixels of numpyFrom, and add these to numpyTo according to the description after equation 11

numpyTo[:, :, 0] = numpyTo[:, :, 0].clip(0.0, 100.0)
numpyTo[:, :, 1] = numpyTo[:, :, 1].clip(-127.0, 127.0)
numpyTo[:, :, 2] = numpyTo[:, :, 2].clip(-127.0, 127.0)


numpyOutput = cv2.cvtColor(src=numpyTo, code=cv2.COLOR_Lab2BGR)

cv2.imwrite(filename='./02-colortransfer.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))