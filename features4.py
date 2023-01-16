import numpy as np
import model
import torch

def prep(points):
    l1 = np.sqrt(np.sum((points[:,0]-np.array([-0.375, 0]))**2,1))
    l2 = np.sqrt(np.sum((points[:,1] - points[:,0])**2,1))
    l3 = np.sqrt(np.sum((points[:,2] - points[:,1])**2,1))
    l4 = np.sqrt(np.sum((points[:,3] - points[:,2])**2,1))
    l5 = np.sqrt(np.sum((points[:,3] - np.array([0.375, 0]))**2,1))

    v1 = points[:,0] - np.array([-0.375, 0])
    v1 = v1.reshape(len(points),2,1)

    v2 = points[:,1] - points[:,0]
    v2 = v2.reshape(len(points),2,1)

    v3 = points[:,2] - points[:,1]
    v3 = v3.reshape(len(points),2,1)

    v4 = points[:,3] - points[:,2]
    v4 = v4.reshape(len(points),2,1)

    p4 = np.zeros([len(points),2])
    p4[:,0] = 0.375
    v5 = p4 - points[:,3]
    v5 = v5.reshape(len(points),2,1)

    A = np.concatenate([v1,v2],2)
    B = np.concatenate([v2,v3],2)
    C = np.concatenate([v3,v4],2)
    D = np.concatenate([v4,v5],2)

    detA = np.linalg.det(A)
    detB = np.linalg.det(B)
    detC = np.linalg.det(C)
    detD = np.linalg.det(D)

    v1dv2 = v1.reshape(len(points),1,2)@v2

    v2dv3 = v2.reshape(len(points),1,2)@v3

    v3dv4 = v3.reshape(len(points),1,2)@v4

    v4dv5 = v4.reshape(len(points),1,2)@v5

    v1dv2=v1dv2.flatten()
    v2dv3 = v2dv3.flatten()
    v3dv4 = v3dv4.flatten()
    v4dv5 = v4dv5.flatten()

    theta1 = np.arctan2(detA,v1dv2)
    theta2 = np.arctan2(detB,v2dv3)
    theta3 = np.arctan2(detC,v3dv4)
    theta4 = np.arctan2(detD,v4dv5)
    L = np.linspace(-3/4,3/4,5)

    c=np.concatenate([points,np.repeat(np.array([[[L[1],0],[L[3],0]]]),points.shape[0],axis=0)],1)

    c[:,[0,4]] = c[:,[4,0]]
    c[:,[1,4]] = c[:,[4,1]]

    L = l1 + l2 + l3 + l4 + l5

    rs = len(points)
    trainer = np.concatenate([(l1/L).reshape(rs,1), ((l1+l2)/L).reshape(rs,1), ((l1+l2+l3)/L).reshape(rs,1),((l1+l2+l3+l4)/L).reshape(rs,1), ((theta1+np.pi)/(2*np.pi)).reshape(rs,1), ((theta2+np.pi)/(2*np.pi)).reshape(rs,1) , ((theta3+np.pi)/(2*np.pi)).reshape(rs,1), ((theta4+np.pi)/(2*np.pi)).reshape(rs,1)],1)

    return trainer,c