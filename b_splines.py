import numpy as np
import matplotlib.pyplot as plt

#Build basis functions and evaluate basis function at t.
#Returns a vector of basis function evaluations - non-zero where t affects the basis function
#and zero where t doesn't affect the  basis function.

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

#Returns the points between 0 and 1 where evaluations occur and matrix containing rows of basis evaluations  
#across all the points in the mesh interval. "Tracing the basis functions."

def trace(knots, mesh, p, clamp):
  num_knots = len(knots)
  num_basis = num_knots - 4
  X = []
  Y = np.zeros((num_basis , mesh))
  first = 0
  last = 0
  if clamp==False:
    array = np.array([])
    
    for index,i in enumerate(np.linspace(0,0.999,mesh)):
      
      if i>=knots[p] and i<=knots[num_knots-p-1]:
        if X==[]:
          first = index
        N = construct(i,knots,3)
        X.append(i)
        array = np.append(array,index)
        for j,k in enumerate(N[0:num_basis]):
          Y[j,index] = k
        last = index
    
    ret = (X,Y[:,first:last+1])
  else:
    for index,i in enumerate(np.linspace(0,0.999,mesh)):  
      N = construct(i,knots,3)
      X.append(i)
      for j,k in enumerate(N[0:num_basis]):
        Y[j,index] = k
    ret = (X,Y)

  return ret

#Plot a tuple.
def plot_basis(T):
  for i in T[1]:
    plt.plot(T[0],i)

#Compute B-spline at points in the mesh interval .
def b_spline(C,knots,mesh,clamp):
  T = trace(knots,mesh,3,clamp)
  Ct = 0 
#Control point i * Basis function i across points in mesh interval + ...
  for i,j in enumerate(T[1]):
    Ct += C[i] * j  
  return Ct

#Plot the control polygon and B-spline.
def plot_b_spline(Ct,C): #Accepts B-spline evaluations, control points.
  plt.scatter(Ct[0],Ct[1])
  plt.plot(list(map(lambda x: x[0][0],C)), list(map(lambda x: x[1][0],C)), c='k')

