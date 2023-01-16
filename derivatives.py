def first_derivative(P, knots): #Accepts the original interpolation points and original set of knots.
  Q=[] #1st derivative control points

  for i in range(len(P)-1):
    Q.append((3/(knots[i+4]-knots[i+1]))*(P[i+1]-P[i]))

  knots = knots[1:-1] #New set of knots the 1st derivative is defined over.

  #C = b_spline(Q,knots,1000,True,2)

  return (Q, knots)

def second_derivative(Q, knots): #Accepts the 1st derivative control points and set of knots the 1st derivative is defined over.
  R=[] #2nd derivative control points

  for i in range(len(Q)-1):
    R.append((2/(knots[i+3]-knots[i+1]))*(Q[i+1]-Q[i]))

  knots = knots[1:-1] #New set of knots the 2nd derivative is defined over.

  #C = b_spline(R,knots,1000,True,1)
  
  return (R, knots)
