import numpy as np
def de_boor(knots, t, CP, p): #Accepts original knots, parameter t to evaluate at, Control points and power.
  k = 0
  s = p+1,CP.shape[1],CP.shape[2]
  E = np.zeros(s)
  while t > knots[p+k+1]:
    k = k+1
  for i in range(p+1):
    E[i] = CP[i+k]
  for j in range(1,p+1):
    for i in range(0,p-j+1):
      E[i] = ((knots[p+1+i+k]-t)*E[i]/(knots[p+1+i+k]-knots[i+k+j])) + ((t-knots[i+k+j])*E[i+1]/(knots[p+1+i+k]-knots[i+k+j]))
  return E[0] #returns the B-splines evaluated at t.