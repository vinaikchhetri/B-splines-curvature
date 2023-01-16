import numpy as np
import model
import torch

def prep(points):
  l1 = np.sqrt(np.sum((points[:,0]-np.array([-0.5, 0]))**2,1))
  l2 = np.sqrt(np.sum((points[:,1] - points[:,0])**2,1))
  l3 = np.sqrt(np.sum((points[:,1] - np.array([0.5, 0]))**2,1))
  v1 = points[:,0] - np.array([-0.5, 0])
  v1 = v1.reshape(len(points),2,1)
  v2 = points[:,1] - points[:,0]
  v2 = v2.reshape(len(points),2,1)
  p4 = np.zeros([len(points),2])
  p4[:,0] = 0.5
  v3 = p4 - points[:,1]
  v3 = v3.reshape(len(points),2,1)
  A = np.concatenate([v1,v2],2)
  B = np.concatenate([v2,v3],2)
  detA = np.linalg.det(A)
  detB = np.linalg.det(B)
  v1dv2 = v1.reshape(len(points),1,2)@v2
  v2dv3 = v2.reshape(len(points),1,2)@v3
  v1dv2=v1dv2.flatten()
  v2dv3 = v2dv3.flatten()
  theta1 = np.arctan2(detA,v1dv2)
  theta2 = np.arctan2(detB,v2dv3)
  c=np.concatenate([points,np.repeat(np.array([[[-0.5,0],[0.5,0]]]),points.shape[0],axis=0)],1)
  c[:,[0,2]] = c[:,[2,0]]
  c[:,[1,2]] = c[:,[2,1]]
  L = l1 + l2 + l3
  rs = len(points)
  trainer = np.concatenate([(l1/L).reshape(rs,1), ((l1+l2)/L).reshape(rs,1), ((theta1+np.pi)/(2*np.pi)).reshape(rs,1), ((theta2+np.pi)/(2*np.pi)).reshape(rs,1)],1)
  return trainer,c