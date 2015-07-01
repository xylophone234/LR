#coding:UTF-8
import numpy as np
import math

epslon=1e-30

def sigmoid(x):
	return 1.0/(1.0+math.exp(-x))

npsigmoid=np.vectorize(sigmoid,otypes=[np.float])

def normal(x):
	y=x*x
	return math.sqrt(y.sum())

# 大于1的梯度归一化，小于1不变
def speedup(x):
	n=normal(x)
	if n>1 :
		return x/n
	else:
		return x

class LR():
	def __init__(self,n):
		self.w=np.zeros((1,n))
		self.b=0

	def predict(self,x):
		z=np.dot(self.w,x)+self.b
		return npsigmoid(z)

	def train(self,trainx,trainy,n,alpha=0.1,pho=0.1):
		i=0
		dw=np.array([1])
		db=np.array([1])
		while(i<n and (normal(dw)>epslon or normal(db)>epslon)):
			i+=1
			t=self.predict(trainx)
			delta=trainy-t
			print (delta**2).sum()
			e=delta*trainx
			dw=e.sum(axis=1)
			# regulary
			# print dw,self.w
			dw=dw-pho*self.w

			db=delta.sum(axis=1)
			self.w+=speedup(dw)*alpha
			self.b+=speedup(db)*alpha

def test():
	x=np.arange(0,10,1).reshape(2,-1)
	y=np.array([[0.1,0.2,0.7,0.7,0.8]])
	lr=LR(2)
	lr.train(x,y,1000,0.02,0.0)
	print 'y=',y
	print 'lr.predict(x)=',lr.predict(x)
	print 'lr.w=',lr.w

if __name__ == '__main__':
	test()