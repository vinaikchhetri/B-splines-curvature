import numpy as np

def construct(t,knots,p):
  N = np.zeros(len(knots)-1)
  lenN = len(N)
  for i,k in enumerate(knots[1:]):
    if t<k and t>= knots[i]:
      N[i] = 1
    else:
      N[i] = 0
  for power in range(1,p+1):

    for index in range(0,lenN-power):
      num = (t-knots[index])
      den = (knots[index+power]-knots[index])
      if den == 0:
        a = 0
      else:
        a = num/ den
     
      num = (knots[index+power+1]-t)
      den = (knots[index+power+1]-knots[index+1])
      if den == 0:
        b = 0
      else:
        b = num/den
      N[index] = N[index]*a + N[index+1]*b

  return N
