# CTR
import numpy as np
import logisticR
a=np.loadtxt('LRTrainNew.txt')
b=np.transpose(a)
x=b[2:,:]
y=b[0:2,:]
lrw=logisticR.LRW(4)
lrw.train(x,y,1000,0.02,0.0)
print lrw.w