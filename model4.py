import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def construct(t,knots,p):  
  N = torch.zeros(knots.shape[0],knots.shape[1]-1,dtype=torch.double).to(device)
  vec = torch.zeros(len(knots),dtype=torch.double).to(device)
  vec2 = torch.ones(len(knots),dtype=torch.double).to(device)

  lenN = N.shape[1] - 1
  for i in range(knots[0,1:].shape[0]):
    v = torch.where(torch.logical_and(t<knots[:,i+1],t>= knots[:,i]),vec2,vec)
    N[:,i] = v
    # print(N[:,i])

  for power in range(1,p+1):
    for index in range(0,lenN-power):
      num = (t-knots[:,index])
      den = (knots[:,index+power]-knots[:,index])
      a = torch.zeros(knots.shape[0],dtype=torch.double).to(device)
      
      ind = torch.nonzero(den, as_tuple=True)
      a[ind] = num[ind]/den[ind]

     
      num = (knots[:,index+power+1]-t)
      den = (knots[:,index+power+1]-knots[:,index+1])
      # if den == 0:
      #   b = torch.tensor([0])
      # else:
      #   b = torch.tensor([num/ den])
      b = torch.zeros(knots.shape[0],dtype=torch.double).to(device)
      
      ind = torch.nonzero(den, as_tuple=True)
      b[ind] = num[ind]/den[ind]
      
      N[:,index] = N[:,index].clone()*a.clone() + N[:,index+1].clone()*b.clone()
  
  
  return N

def construct_A(P, knots): #accepts Points to interpolate and the knots.

  for i in range(knots[0,3:-3].shape[0]):
 
    a = construct(knots[:,i+3],knots,3)
    if knots[0,i+3]==1:
      a = torch.zeros(knots.shape[0], knots.shape[1]-1).to(device)
      a[:,len(P[0])+1]=1
      

    if i==0:
      A = torch.unsqueeze(a,1)
      #A=torch.cat((A,torch.unsqueeze(a,0)),1) 
    else:
      A = torch.cat((A,torch.unsqueeze(a,1)),1) 

      #A=torch.cat((A,torch.unsqueeze(a,0)),0) 
  A = A[:,:,0:len(P[0])+2]

  a = torch.zeros(knots.shape[0], len(P[0])+2).to(device)

  A = torch.cat((torch.unsqueeze(a,1),A),1) 
  A = torch.cat((A,torch.unsqueeze(a,1)),1) 
  A[:,0,0] += knots[:,5]
  A[:,0,1] += -knots[:,4]-knots[:,5]
  A[:,0,2] += knots[:,4]


  # #Last row is C''(u_n) = 0

  A[:,-1,-3] += 1-knots[:,len(P[0])+1]
  A[:,-1,-2] += knots[:,len(P[0])+1] + knots[:,len(P[0])]-2
  A[:,-1,-1] += 1 - knots[:,len(P[0])]
  
  return A

def solve(P, knots):
  L = (list(map(lambda x: x[:,0],P)))
  count = 0
  for i in L:
    if count == 0:
      Px = i.unsqueeze(0)
    else:
      Px = torch.cat((Px,i.unsqueeze(0)),0)
    count+=1
  
  Px = torch.cat((torch.zeros(Px.shape[0],1,1).to(device), Px),1)
  Px = torch.cat((Px, torch.zeros(Px.shape[0],1,1).to(device)),1)

  L = (list(map(lambda x: x[:,1],P)))
  count = 0
  for i in L:
    if count == 0:
      Py = i.unsqueeze(0)
    else:
      Py = torch.cat((Py,i.unsqueeze(0)),0)
    count+=1
  Py = torch.cat((torch.zeros(Py.shape[0],1,1).to(device), Py),1)
  Py = torch.cat((Py, torch.zeros(Py.shape[0],1,1).to(device)),1)

  #Py[:,0] = Py[:,1]
  #Py[:,1] = 0

  #Py[:,-1] = Py[:,-2]
  #Py[:,-2] = 0

  A = construct_A(P, knots)
  
  inv,info = torch.linalg.inv_ex(A)
  Cx = torch.bmm(inv, Px)
  Cy = torch.bmm(inv, Py)

  D = torch.cat((Cx,Cy),2)
  D = torch.unsqueeze(D,3)

  return D
  
def de_boor(knotss, t, CP, p): #Accepts original knots, parameter t to evaluate at, Control points and power.
  k = 0
  s = CP.shape[0],p+1,CP.shape[2],CP.shape[3]
  E = torch.zeros(s,dtype=torch.double).to(device)
  vec = torch.zeros(len(knotss),dtype=torch.double).to(device)
  vec2 = torch.ones(len(knotss),dtype=torch.double).to(device)
  k = torch.zeros(len(knotss),dtype=torch.double).to(device)

  for i in range(p+1,len(knotss[0])):
    v = torch.where(t>knotss[:,i],vec2,vec)
    k += v

  
  for i in range(p+1):
    I = torch.fill(v,i).to(device)
    # print(I)
    # print(I+k)
    # print(CP[torch.arange(len(knotss)),(I+k).type(torch.LongTensor)])
    # print()
    # break
    E[torch.arange(len(knotss)),I.type(torch.LongTensor)] = CP[torch.arange(len(knotss)),(I+k).type(torch.LongTensor)]
  
  
  al = torch.arange(len(knotss))
  for j in range(1,p+1):
    for i in range(0,p-j+1):
      I = torch.fill(v,i)
      P = torch.fill(v,p)
      J = torch.fill(v,j)
      indI = I.type(torch.LongTensor)
      indPIK1 = (P+1+I+k).type(torch.LongTensor)
      indIKJ = (I+k+J).type(torch.LongTensor)
      indI1 = (I+1).type(torch.LongTensor)
      ans1 = torch.cat(((knotss[al,indPIK1 ]).unsqueeze(1),(knotss[al,indPIK1 ]).unsqueeze(1)),1).unsqueeze(2)
      ans2 = torch.cat(((knotss[al,indIKJ ]).unsqueeze(1),(knotss[al,indIKJ ]).unsqueeze(1)),1).unsqueeze(2)
      tt = torch.cat(((t).unsqueeze(1),(t).unsqueeze(1)),1).unsqueeze(2)    
      #E = E.type(torch.DoubleTensor)
      E[al,indI] = (((ans1-tt)*E[al,indI].clone()/(ans1- ans2)) 
      + ((tt-ans2)*E[al,indI1].clone()
      /(ans1-ans2)))


  return E[:,0] #returns the B-splines evaluated at t.
  # return 

def first_derivative(P, knotss): #Accepts the original interpolation points and original set of knots.
  #Q=[] #1st derivative control points
  Q = torch.zeros((len(P),len(P[0])-1,2,1),dtype=torch.double).to(device)
  for i in range(len(P[0])-1):
    ans1 = torch.cat(((knotss[:,i+4]-knotss[:,i+1]).unsqueeze(1), (knotss[:,i+4]-knotss[:,i+1]).unsqueeze(1)),1).unsqueeze(2)
    Q[:,i] = ((3/ans1)*(P[:,i+1]-P[:,i]))

  #knots = knots[1:-1] #New set of knots the 1st derivative is defined over.
  k = knotss[:,1:-1]
  #C = b_spline(Q,knots,1000,True,2)

  return (Q, k)

def second_derivative(Q, knotss): #Accepts the 1st derivative control points and set of knots the 1st derivative is defined over.
  #R=[] #2nd derivative control points
  
  R = torch.zeros((len(Q),len(Q[0])-1,2,1),dtype=torch.double).to(device)
  #print(Q)
  for i in range(len(Q[0])-1):
    ans1 = torch.cat(((knotss[:,i+3]-knotss[:,i+1]).unsqueeze(1), (knotss[:,i+3]-knotss[:,i+1]).unsqueeze(1)),1).unsqueeze(2)
    #print(((2/ans1)*(Q[:,i+1]-Q[:,i])))


    R[:,i] = ((2/ans1)*(Q[:,i+1]-Q[:,i]))

  # knots = knots[1:-1] #New set of knots the 2nd derivative is defined over.
  k = knotss[:,1:-1]
  #C = b_spline(R,knots,1000,True,1)
  
  return (R, k)

def local_curvature(C, knotss, t, p):
  (Q, dknots) = first_derivative(C, knotss) 
  (R, ddknots) = second_derivative(Q, dknots)

  spline_prime = de_boor(dknots, t, Q, p-1)
  spline_dprime = de_boor(ddknots, t, R, p-2)
  # print(spline_prime)
  # print(spline_prime[:,0,0])
  curvature = ((spline_prime[:,0,0]*spline_dprime[:,1,0])-(spline_prime[:,1,0]*spline_dprime[:,0,0]))/((spline_prime[:,0,0]**2+spline_prime[:,1,0]**2)**(3))**(0.5)

  return curvature 

def global_curvature(C, knotss, p, num_pts): #Accepts control points, original knots and power.
 L = torch.linspace(knotss[0,p].item(), knotss[0,len(knotss[0])-(p-1)].item(), num_pts)
 h = 1.0/(len(L)-1)

 S1 = 0
 S2 = 0
 #s1=0
 #s2=0
 (Q, knots2) = first_derivative(C, knotss) 
#  curvature = local_curvature(C, knotss, 0.1, 3)
#  return curvature 
 v = torch.ones(len(knotss),dtype=torch.double)
 for i,j in enumerate(L):
  J = torch.fill(v,j).to(device)
  if i%2 == 1: 
   curvature = local_curvature(C, knotss, J, 3)
   evl = de_boor(knots2,J,Q,2)
   S1 +=  (curvature**2) * torch.flatten(torch.norm(evl,dim=1))
  #  print('curvature**2',curvature**2)
  #  print('torch.norm(evl)',torch.norm(evl,dim=1))
  #  print('flatten',(curvature**2) * torch.flatten(torch.norm(evl,dim=1)))
  #  break
  else:
   curvature = local_curvature(C, knotss, J, 3)
   evl = de_boor(knots2,J,Q,2)
   S2 += (curvature**2) * torch.flatten(torch.norm(evl,dim=1))

 integration = (h/3)*(4*S1 + 2*S2)

	
 return integration
 #+(lambda_var)*arc

def arc(C, knots, p, num_pts): #Accepts control points, original knots and power.
 L = torch.linspace(knots[0,p].item(), knots[0,len(knots[0])-(p-1)].item(), num_pts)
  # L = torch.linspace(knotss[0,p].item(), knotss[0,len(knotss[0])-(p-1)].item(), num_pts)
 h = 1.0/(len(L)-1)
 s1=0
 s2=0
 (Q, knots2) = first_derivative(C, knots) 
 v = torch.ones(len(knots),dtype=torch.double)

 for i,j in enumerate(L):
  J = torch.fill(v,j)
  if i%2 == 1:
   evl = de_boor(knots2,J,Q,2)
   s1 += torch.flatten(torch.norm(evl,dim=1))
  else:
   evl = de_boor(knots2,J,Q,2) 
   s2 += torch.flatten(torch.norm(evl,dim=1))

 arc = (h/3)*(4*s1 + 2*s2)
	
 return arc


class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(8,128,dtype=torch.double)
        self.lin2 = nn.Linear(128,128,dtype=torch.double)
        #self.lin3 = nn.Linear(128,128,dtype=torch.double)
        self.lin3 = nn.Linear(128,5,dtype=torch.double)  
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0.01)  

        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0.01) 

        nn.init.xavier_uniform_(self.lin3.weight)
        self.lin3.bias.data.fill_(0.01)  

    def forward(self, input):
        x = input[0]
   
        #print(x.shape)
        # x = F.leaky_relu((self.lin1(x)),negative_slope=0.8)
        # x = F.leaky_relu(self.lin2(x),negative_slope=0.8)
        x = F.relu((self.lin1(x)))
        x = F.relu(self.lin2(x))
        #x = (F.relu(self.lin3(x)))
        x = ((self.lin3(x)))

        #x = torch.sigmoid(x)
        # x, indices = torch.sort(x.clone(),1)
        #print(x.shape)
        x = nn.Softmax(dim=1)(x)
        w = torch.zeros((x.shape[0],x.shape[1]-1))
        for i in range(x.shape[1]-1):
          # print(x.shape)
          # print(x[:,i+1].shape)
          # print(w[:,i].shape)
          if i==0:
            w[:,i] =  x[:,i]
          else:
            w[:,i] = x[:,i] + w[:,i-1].clone().detach()


        #w, indices = torch.sort(x.clone(),1)
        # z = x[:,0].clone()+x[:,1].clone()
        # y = x[:,1].clone().detach() + x[:,2] +  x[:,3]
        # w = torch.cat([z.unsqueeze(1),y.unsqueeze(1)],dim=1)
        #x = torch.cumsum(x , dim=1)
        #x = x[:,:-1]

        
        knots_left = torch.zeros((w.shape[0],8),dtype=torch.double).to(device=device)
        knots_right = torch.ones((w.shape[0],4),dtype=torch.double).to(device=device)
        knots_left[:,4:8] = knots_left[:,4:8].clone() + w
        knots = torch.cat([knots_left, knots_right], dim = 1)
        print(knots[0:5])

        # P = input[1]
        # C = solve(P, knots)
        # curv = global_curvature(C, knots, 3, 1000)
        #arc1 = arc(C,knots,3,1000)


        # return curv,knots


        return knots


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = Net1()

    def forward(self, input):
        knots = self.first(input)
        # x, indices = torch.sort(x.clone(),1)

        # ig1 = (knots[:,5] >= 1).nonzero(as_tuple=True)
        
        # if not ig1[0].nelement() == 0: 
        #   offset1 = 1 - 3*knots[ig1[0],5].clone()/2 
        #   knots[ig1[0],5] = knots[ig1[0],5].clone() + offset1 
        
        P = input[1]
        C = solve(P, knots)
        curv = global_curvature(C, knots, 3, 1000)
        #arc1 = arc(C,knots,3,1000)


        return curv,knots
        #return arc1,knots



