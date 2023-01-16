import numpy as np
import deBoor
import derivatives
from numpy import linalg as LA

def arc(C, knots, p, num_pts): #Accepts control points, original knots and power.
 L = np.linspace(knots[p], knots[len(knots)-(p-1)], num_pts)
 h = 1.0/(len(L)-1)
 s1=0
 s2=0
 (Q, knots2) = derivatives.first_derivative(C, knots) 
 Q = np.asarray(Q)
 for i,j in enumerate(L):
  if i%2 == 1:
   evl = deBoor.de_boor(knots2,j,Q,2)
   s1+=LA.norm(evl)
  else:
   evl = deBoor.de_boor(knots2,j,Q,2) 
   s2+=LA.norm(evl)

 arc = (h/3)*(4*s1 + 2*s2)
	
 return arc