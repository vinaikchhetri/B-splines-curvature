import numpy as np
import b_splines

#Constructs the interpolation matrix for B-splines with natural conditions.
def construct_A(P, knots): #accepts Points to interpolate and the knots.
	A=[[]]
#tri-diagonal part of the matrix.
	for i,k in enumerate(knots[3:-3]):
	  a = b_splines.construct(k,knots,3)
	  if k==1:
	    a = np.zeros(len(knots)-1)
	    a[len(P)+1]=1
	  if i==0:
	    A=np.concatenate((A,[a]),1) 
	  else:
	    A=np.concatenate((A,[a]),0) 
	A = A[:,0:len(P)+2]

#First row is C''(u_0) = 0
	a=np.zeros(len(P)+2)
	a[0]=knots[5]
	a[1]=-knots[4]-knots[5]
	a[2]=knots[4]
	A=np.concatenate(([a],A),0) 

#Last row is C''(u_n) = 0
	a=np.zeros(len(P)+2)
	a[-3]=1-knots[len(P)+1]
	a[-2]=knots[len(P)+1]+knots[len(P)]-2
	a[-1]=1-knots[len(P)]


	A=np.concatenate((A,[a]),0) 

	return A

#Solve for control points via A-1@P
def solve(P, knots): #accepts Points to interpolate and the knots.
	
#x-coordinates of the points	
	Px = (list(map(lambda x: x[0][0],P)))
	Px=np.concatenate(([0],np.array(Px)),0)
	Px=np.concatenate((np.array(Px),[0]),0)

#y-coordinates of the points
	Py = (list(map(lambda x: x[1][0],P)))
	Py=np.concatenate(([0],np.array(Py)),0)
	Py=np.concatenate((np.array(Py),[0]),0)

#x-coordinates and y-coordinates of the control points.
	A = construct_A(P, knots)
	Cx = np.dot(np.linalg.inv(A),np.expand_dims(Px,1))
	Cy = np.dot(np.linalg.inv(A),np.expand_dims(Py,1))

#concatenate x-coordinates and y-coordinates.
	D=np.concatenate((Cx,Cy),1)
	D=np.expand_dims(D,2)

	return D

