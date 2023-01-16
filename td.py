from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import torch
import model



#c = np.array([[[0],[0]],[[2.5],[1]],[[2.75],[4]], [[3],[5]] ])
def tdplot(c):
    c = torch.tensor(c).unsqueeze(0)
    D = torch.linspace(0,1,51)
    D = D[1:-1]
    K = []
    for i,j in enumerate(D):
        for k in D[i+1:]:
            K.append((j,k))
    Kc = torch.tensor(K)

    f = torch.zeros(1176,4)
    s = torch.ones(1176,4)
    knots = torch.cat([f,Kc],dim=1)
    knots = torch.cat([knots,s],dim=1)

    c = c.repeat(1176,1,1,1)
    c = c.to(torch.double)
    knots = knots.to(torch.double)
    C = model.solve(c,knots)
    curv = model.global_curvature(C, knots,3,1000)

    x = np.linspace(0, 1, 51)
    y = np.linspace(0, 1, 51)
    x=x[1:-1]
    y=y[1:-1]
    X, Y = np.meshgrid(x, y)

    Z = np.zeros((X.shape[0],X.shape[1]))
    curv = np.asarray(curv)
    co=0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if i<j and co<=1175:
                Z[i][j] = np.log10(curv[co])
                co+=1
            else:
                Z[i][j] = 'nan'
    return (X,Y,Z)



