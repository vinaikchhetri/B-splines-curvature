import numpy as np
import deBoor
import derivatives
import math
from numpy import linalg as LA

# def global_curvature(R, knots, p):
# 	L = np.linspace(knots[p], knots[len(knots)-(p+1)], 10005)
# 	h = 1.0/(len(L)-1)

# 	S1 = 0
# 	S2 = 0

# 	for i,j in enumerate(L[1:-1]):
# 		if i%2 == 1:
# 			a = deBoor.de_boor(knots, j, R, p)
# 			S1 += np.dot(a.T,a) 
# 		else:
# 			a = deBoor.de_boor(knots, j, R, p)
# 			S2 += np.dot(a.T,a)   

# 	integration = (h/3)*(4*S1 + 2*S2)
# 	return integration

#Returns norm-squared of the curvature at point t.
def local_curvature(C, knots, t, p):

	(Q, dknots) = derivatives.first_derivative(C, knots) 
	(R, ddknots) = derivatives.second_derivative(Q, dknots)
	Q = np.asarray(Q)
	spline_prime = deBoor.de_boor(dknots, t, Q, p-1)
	R = np.asarray(R)
	spline_dprime = deBoor.de_boor(ddknots, t, R, p-2)
	curvature = ((spline_prime[0][0]*spline_dprime[1][0])-(spline_prime[1][0]*spline_dprime[0][0]))/((spline_prime[0][0]**2+spline_prime[1][0]**2)**(3))**(0.5)
	#print("local_curvature: ", curvature)
	return curvature 

def global_curvature(C, knots, p): #Accepts control points, original knots and power.
 L = np.linspace(knots[p], knots[len(knots)-(p-1)], 1000)
 
 h = 1.0/(len(L)-1)
 S1 = 0
 S2 = 0
 (Q, knots2) = derivatives.first_derivative(C, knots)
 Q = np.asarray(Q) 
 
 for i,j in enumerate(L):
  if i%2 == 1:
   curvature = local_curvature(C, knots, j, 3)

   evl = deBoor.de_boor(knots2,j,Q,2)
   S1 +=  LA.norm(evl)*(curvature**2)
  else:
   curvature = local_curvature(C, knots, j, 3)

   evl = deBoor.de_boor(knots2,j,Q,2)
   S2 += LA.norm(evl)*(curvature**2)  

 integration = (h/3)*(4*S1 + 2*S2)
	
 return integration

# def global_curvature(C, knots, p): #Accepts control points, original knots and power.


# 	L = np.linspace(knots[p], knots[len(knots)-(p-1)], 1000)

# 	h = 1.0/(len(L)-1)

# 	S1 = 0
# 	S2 = 0
	
# 	for i,j in enumerate(L):
# 		if i%2 == 1:
# 			curvature = local_curvature(C, knots, j, 3)
# 			#a = deBoor.de_boor(ddknots, j, R, p-2) #Evalutate 2nd derivative of B-spline at j.
# 			#b = a/np.sqrt(np.dot(a.T,a))
# 			#print(a)
# 			#print(np.sqrt(np.dot(a.T,a)))
# 			#print(b)
# 			#print((np.dot(a.T,a)))
# 			#print(np.sqrt(np.dot(a.T,a)))
# 			#print(a/np.sqrt(np.dot(a.T,a)))
# 			#b = a/(np.dot(a.T,a))
# 			S1 +=  curvature**2
# 		else:
# 			curvature = local_curvature(C, knots, j, 3)
# 			#a = deBoor.de_boor(ddknots, j, R, p-2) #Evalutate 2nd derivative of B-spline at j.
# 			#b = a/np.sqrt(np.dot(a.T,a))
# 			#print((np.dot(a.T,a)))
# 			#print(np.sqrt(np.dot(a.T,a)))
# 			#print(a/np.sqrt(np.dot(a.T,a)))
# 			#b = a/(np.dot(a.T,a))
# 			S2 += curvature**2  

# 	integration = (h/3)*(4*S1 + 2*S2)
	
# 	return integration


