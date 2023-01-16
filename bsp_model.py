import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import b_splines
import curvature
import interpolation
import deBoor

from features import prep
from model import Net2
import torch

from td import tdplot

def centripetal(P):
  shifted = np.roll(P,1,axis=0)
  shifted[0] = np.array([[0],[0]])
  sides = P - shifted
  sides = sides[1:]
  d = np.zeros(len(sides))
  for j,i in enumerate(sides):
    d[j] = np.sqrt(np.sqrt(i.T@i)[0][0])
  L = sum(d)
  knots = np.zeros(len(P)+6)
  knots[-4:] = np.ones(4)
  s=0
  for i in range(4,4+len(P)-2):
    s+= d[i-4]
    knots[i] = s/L
  return knots

net2 = Net2()
net2.load_state_dict(torch.load('net-489-198-tracked.pt'))
# net2.load_state_dict(torch.load('net-6466.pt'))
# net2.load_state_dict(torch.load('1st-4d.pt'))
#net2.load_state_dict(torch.load('net-2-tracked-part3.pt'))

net = net2.first

P = np.array([[[-0.5],[0]],[[0],[0.2]],[[0.2],[0.4]],[[0.5],[0]] ])
#P = np.array([[[-0.375],[0]],[[-0.2],[0.2]], [[-0.1],[0.3]],[[0],[0.1]], [[0.1],[0.1]],[[0.2],[0.1]],[[0.375],[0]] ])
#P = np.array([[[-0.375],[0]],[[-0.2],[0.2]], [[-0.1],[0.3]],[[0],[0.1]], [[0.2],[0.1]],[[0.375],[0]] ])

print(P.shape)


trainer,c = prep(P[1:3].reshape(1,2,2))
#trainer,c = prep(P[1:6].reshape(1,5,2))
# trainer,c = prep(P[1:5].reshape(1,4,2))

c=torch.tensor(c)
c.type(torch.DoubleTensor)
c = torch.unsqueeze(c,3)
trainer = torch.tensor(trainer)
trainer.type(torch.DoubleTensor)

cknots = centripetal(P)
cC = interpolation.solve(P, cknots)
clamp = True
cF = b_splines.b_spline(cC,cknots,1000,clamp)
cinte = float(curvature.global_curvature(cC, cknots, 3))
cinte_org = cinte 


knots = net((trainer,c))
knots = knots[0].detach().numpy()


C = interpolation.solve(P, knots)
L = []
for t in knots[3:-3]:
   L.append(deBoor.de_boor(knots, t, C, 3))
print(L)


#Controls points
#C = np.array([[[0],[2]],[[1],[2.5]],[[1.5],[0.5]],[[1.7],[0.6]], [[1.9],[0.9]]]) #4 
#C = np.array([[[0],[2]],[[1],[2.5]],[[1.5],[0.5]],[[1.7],[0.6]], [[1.9],[0.9]]]) #5 
# Cs = np.copy(C) changed 3
Ps = np.copy(P) #changed 3

clamp = True
F = b_splines.b_spline(C,knots,1000,clamp)
inte = float(curvature.global_curvature(C, knots, 3))
inte_org = inte 
#figure.subplot.right
mpl.rcParams['figure.subplot.right'] = 0.8

#set up a plot
fig,axes = plt.subplots(1,1,figsize=(9.0,8.0),sharex=clamp)
ax1 = axes



pind = None #active point
epsilon = 10 #max pixel distance

def update():
    global P #chamged 4
    global F
    global C


    # trainer,c = prep(P[1:5].reshape(1,4,2))
    trainer,c = prep(P[1:3].reshape(1,2,2))
    #trainer,c = prep(P[1:6].reshape(1,5,2))

    c=torch.tensor(c)
    c.type(torch.DoubleTensor)
    c = torch.unsqueeze(c,3)
    trainer = torch.tensor(trainer)
    trainer.type(torch.DoubleTensor)

    knots = net((trainer,c))
    knots = knots[0].detach().numpy()
    print(knots)

    C = interpolation.solve(P, knots) #changed 5
    inte = float(curvature.global_curvature(C, knots, 3))

    cknots = centripetal(P)
    cC = interpolation.solve(P, cknots)
    clamp = True
    cF = b_splines.b_spline(cC,cknots,1000,clamp)
    cinte = float(curvature.global_curvature(cC, cknots, 3))

    text4.set_text(str(cinte))
    cm.set_ydata(cF[1]) #Update spline curve
    cm.set_xdata(cF[0])
  

    text2.set_text(str(inte))
    ptsl.set_ydata(np.asarray(list(map(lambda x: x[1][0],P)))) #Update points
    ptsl.set_xdata(np.asarray(list(map(lambda x: x[0][0],P)))) #Update points
    l.set_ydata(np.asarray(list(map(lambda x: x[1][0],C)))) #Update control points
    l.set_xdata(np.asarray(list(map(lambda x: x[0][0],C)))) #Update control points
    pl.set_ydata(np.asarray(list(map(lambda x: x[1][0],C)))) #Update polygon line
    pl.set_xdata(np.asarray(list(map(lambda x: x[0][0],C))))  #Update polygon line
    F = b_splines.b_spline(C,knots,1000,clamp)
    m.set_ydata(F[1]) #Update spline curve
    m.set_xdata(F[0])

    # redraw canvas while idle
    fig.canvas.draw_idle()
   

def reset(event):
    global P
    global C
    global F


    text2.set_text(str(inte_org))
    # C = np.copy(Cs) #changed 6
    P = np.copy(Ps) #changed 6

    trainer,c = prep(P[1:3].reshape(1,2,2))
    # trainer,c = prep(P[1:5].reshape(1,4,2))
    #trainer,c = prep(P[1:6].reshape(1,5,2))

    c=torch.tensor(c)
    c.type(torch.DoubleTensor)
    c = torch.unsqueeze(c,3)
    trainer = torch.tensor(trainer)
    trainer.type(torch.DoubleTensor)

    knots = net((trainer,c))
    knots = knots[0].detach().numpy()

    C = interpolation.solve(P, knots) #changed 6

    cknots = centripetal(P)
    cC = interpolation.solve(P, cknots)
    clamp = True
    cF = b_splines.b_spline(cC,cknots,1000,clamp)

    text4.set_text(str(cinte_org))
    cm.set_ydata(cF[1]) #Update spline curve
    cm.set_xdata(cF[0])

    ptsl.set_ydata(np.asarray(list(map(lambda x: x[1][0],P)))) #Update points
    ptsl.set_xdata(np.asarray(list(map(lambda x: x[0][0],P)))) #Update points

    l.set_ydata(np.asarray(list(map(lambda x: x[1][0],C)))) #Reset control points
    l.set_xdata(np.asarray(list(map(lambda x: x[0][0],C)))) #Reset control points
    pl.set_ydata(np.asarray(list(map(lambda x: x[1][0],C)))) #Reset polygon line
    pl.set_xdata(np.asarray(list(map(lambda x: x[0][0],C)))) #Reset polygon line
    #Reset Spline
    F = b_splines.b_spline(C,knots,1000,clamp)
    m.set_ydata(F[1])
    m.set_xdata(F[0])


    # ax2.plot_surface(Xc, Yc, Zc, rstride=1, cstride=1,
    #             cmap='viridis', edgecolor='none')
    # ax2.set_title('surface')

    # th = ax2.plot_surface(Y, X, Z, rstride=1, cstride=1,
    #                 cmap='viridis', edgecolor='none')
    # th2 = ax2.scatter(knots[4],knots[5],np.log10(inte),c='k') #mod
    # th3 = ax2.scatter(cknots[4],cknots[5],np.log10(cinte),c='b') #cent


    # redraw canvas while idle
    fig.canvas.draw_idle()


def button_press_callback(event):
    'whenever a mouse button is pressed'
    global pind
    if event.inaxes is None:
        return
    if event.button != 1:
        return
    
    pind = get_ind_under_point(event)
    print("pind ",pind)    

def button_release_callback(event):
    'whenever a mouse button is released'
    global pind
    if event.button != 1:
        return
    pind = None

def get_ind_under_point(event):
    'get the index of the vertex under point if within epsilon tolerance'

    # display coords
    #print('display x is: {0}; display y is: {1}'.format(event.x,event.y))
    t = ax1.transData.inverted()
    tinv = ax1.transData #some sort of matrix

    xy = t.transform([event.x,event.y])
    #print('data x is: {0}; data y is: {1}'.format(xy[0],xy[1]))
    xvals = np.asarray(list(map(lambda x: x[0][0],P))) #changed 1
    yvals = np.asarray(list(map(lambda x: x[1][0],P))) #changed 1
    xr = np.reshape(xvals,(np.shape(xvals)[0],1))
    yr = np.reshape(yvals,(np.shape(yvals)[0],1))
    xy_vals = np.append(xr,yr,1)
    xyt = tinv.transform(xy_vals)
    xt, yt = xyt[:, 0], xyt[:, 1]
    d = np.hypot(xt - event.x, yt - event.y)
    indseq, = np.nonzero(d == d.min())
    ind = indseq[0]

    #print(d[ind])
    if d[ind] >= epsilon:
        ind = None
    
    #print(ind)
    return ind

def motion_notify_callback(event):
    'on mouse movement'
    global yvals

    if pind is None:
        return
    if event.inaxes is None:
        return
    if event.button != 1:
        return
    
    #update yvals
    #print('motion x: {0}; y: {1}'.format(event.xdata,event.ydata))
    P[pind][1][0] = event.ydata  #changed 2
    P[pind][0][0] = event.xdata  #changed 2
    update()

    # update curve via sliders and draw
    #sliders[pind].set_val(yvals[pind])
    fig.canvas.draw_idle()



#print(list(map(lambda x: x[0][0],C)))
#print(list(map(lambda x: x[1][0],C)))
pl, =  ax1.plot (list(map(lambda x: x[0][0],C)), list(map(lambda x: x[1][0],C)), 'k--' ,label='Control line') #original curve
# ptsl = ax1.scatter (list(map(lambda x: x[0][0],P)), list(map(lambda x: x[1][0],P)), marker = 'x', c='k', s=np.ones(len(P))*64) #original points #changed 5
ptsl, = ax1.plot (list(map(lambda x: x[0][0],P)),list(map(lambda x: x[1][0],P)),color='k',linestyle='none',marker='x',markersize=8) #original points #changed 55
l, = ax1.plot (list(map(lambda x: x[0][0],C)),list(map(lambda x: x[1][0],C)),color='k',linestyle='none',marker='o',markersize=8) #Just the control points
m, = ax1.plot (F[0], F[1], 'r-', label='model B-Spline') #Curve that will be updated.
cm, = ax1.plot (cF[0], cF[1], 'b-', label='centripetal B-Spline') #Curve that will be updated.
#p_i, = ax1.plot (list(map(lambda x: x[0][0],P)),list(map(lambda x: x[1][0],P)),color='b',linestyle='none',marker='x',markersize=10) #Just the control points


# th = ax2.plot_surface(Y, X, Z, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')





#ax1.set_yscale('linear')
#ax1.set_xlim(0, 2)
#ax1.set_ylim(0,3)
#ax1.set_xlim(-5, 5)
#ax1.set_ylim(-5,5)
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1,1)

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(clamp)
ax1.yaxis.grid(clamp,which='minor',linestyle='--')
ax1.legend(loc=2,prop={'size':22})

# axres = plt.axes([0.84, 0.8-((4)*0.05), 0.12, 0.02])

# bres = Button(axres, 'Reset')
# bres.on_clicked(reset)

# text1 = ax1.text(2.1, 2.5, 'boxed', style='italic',
#         bbox={'facecolor': 'red', 'alpha': 0.4, 'pad': 4})
# text1.set_text('Curvature:')

# text2 = ax1.text(2.1, 2.4, 'boxed', style='italic',
#         bbox={'facecolor': 'red', 'alpha': 0, 'pad': 4})
# text2.set_text(str(inte))

# text1 = ax1.text(5.5, 3.9, 'boxed', style='italic',
#         bbox={'facecolor': 'red', 'alpha': 0.4, 'pad': 4})
# text1.set_text('Curvature:')

# text2 = ax1.text(5.5, 3.5, 'boxed', style='italic',
#         bbox={'facecolor': 'red', 'alpha': 0, 'pad': 4})
# text2.set_text(str(inte))

axres = plt.axes([0.84, 0.8-((4)*0.05), 0.12, 0.02])
bres = Button(axres, 'Reset')
bres.on_clicked(reset)

text1 = ax1.text(0.82, 0.72, "model Curvature", transform=fig.transFigure)
text2 = ax1.text(0.82, 0.7, str(inte), transform=fig.transFigure)

text3 = ax1.text(0.82, 0.68, "cent Curvature", transform=fig.transFigure)
text4 = ax1.text(0.82, 0.66, str(cinte), transform=fig.transFigure)

fig.canvas.mpl_connect('button_press_event', button_press_callback)
fig.canvas.mpl_connect('button_release_event', button_release_callback)
fig.canvas.mpl_connect('motion_notify_event', motion_notify_callback)

plt.show()

