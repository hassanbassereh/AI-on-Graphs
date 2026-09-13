

import numpy as np
import matplotlib.pyplot as plt

# Synthetic data preparation
noise=7
xdata=range(20)
ydata=[3*i+np.random.uniform(-noise,noise) for i in xdata]



plt.figure(figsize=(8,25))
plt.subplot(6,1,1)
# Plotting Sine Graph
plt.scatter(xdata, ydata, color='green')
plt.ylabel('y(Scores)',fontsize=12)
plt.xlabel('x(hours)',fontsize=12)


def Errorfunc(beta):
    error=[]
    for i in range(len(ydata)):
        yminyfit= (ydata[i]-beta*xdata[i])**2
        error.append(yminyfit)
    return np.mean(error)

Beta=np.arange(-12,17+0.05,0.005)
Error= [Errorfunc(j) for j in Beta]

plt.subplot(6,1,2)
plt.plot(Beta,Error)
plt.ylabel('Error(MSE)',fontsize=12)
plt.xlabel(r'$\beta$',fontsize=12)


plt.subplot(6,1,3)
learningrate = 0.00001


betai = -10
B=[]
for q in range(150):
    B.append(betai)
    if q==0:
        plt.scatter(q,betai,c='r',s=5)
    else:
        plt.scatter(q,betai,c='b',s=5)
    ypred = [betai*z for z in xdata]
    yy = [-ypred[i]+ydata[i] for i in range(len(ydata))]
    min2x = [-2*xx for xx in xdata]
    gradient = sum([min2x[i]*yy[i] for i in range(len(yy))])
    betai = betai - learningrate*gradient
print(betai)
plt.text(60,-5,r'$\eta=1*10^{-5}$',fontsize=15)

plt.ylabel(r'$\beta$',fontsize=12)
plt.xlabel('epoch',fontsize=12)


plt.subplot(6,1,4)
learningrate = 0.00001
betai = 10
B=[]
for q in range(150):
    B.append(betai)
    if q==0:
        plt.scatter(q,betai,c='r',s=5)
    else:
        plt.scatter(q,betai,c='b',s=5)
    ypred = [betai*z for z in xdata]
    yy = [-ypred[i]+ydata[i] for i in range(len(ydata))]
    min2x = [-2*xx for xx in xdata]
    gradient = sum([min2x[i]*yy[i] for i in range(len(yy))])
    betai = betai - learningrate*gradient
print(betai)

plt.text(60,5,r'$\eta=1*10^{-5}$',fontsize=15)

plt.ylabel(r'$\beta$',fontsize=12)
plt.xlabel('epoch',fontsize=12)




plt.subplot(6,1,5)
# learningrate = 0.0001
# learningrate = 0.0004
learningrate = 0.0004
betai = 10
B=[]
for q in range(250):
    B.append(betai)
    if q==0:
        plt.scatter(q,betai,c='r',s=5)
    else:
        plt.scatter(q,betai,c='b',s=5)

    ypred = [betai*z for z in xdata]
    yy = [-ypred[i]+ydata[i] for i in range(len(ydata))]
    min2x = [-2*xx for xx in xdata]
    gradient = sum([min2x[i]*yy[i] for i in range(len(yy))])
    betai = betai - learningrate*gradient
print(betai)

plt.text(150,5,r'$\eta=4.0*10^{-4}$',fontsize=15)

plt.ylabel(r'$\beta$',fontsize=12)
plt.xlabel('epoch',fontsize=12)



plt.subplot(6,1,6)

learningrate = 0.00041
betai = 10
B=[]
for q in range(150):
    B.append(betai)
    if q==0:
        plt.scatter(q,betai,c='r',s=5)
    else:
        plt.scatter(q,betai,c='b',s=5)

    ypred = [betai*z for z in xdata]
    yy = [-ypred[i]+ydata[i] for i in range(len(ydata))]
    min2x = [-2*xx for xx in xdata]
    gradient = sum([min2x[i]*yy[i] for i in range(len(yy))])
    betai = betai - learningrate*gradient
print(betai)
plt.text(60,150,r'$\eta=4.1*10^{-4}$',fontsize=15)

plt.ylabel(r'$\beta$',fontsize=12)
plt.xlabel('epoch',fontsize=12)

plt.savefig('lr1-1.png', dpi=200, bbox_inches="tight")

plt.show()



plt.figure(figsize=(5,9))

Beta=np.arange(-12,17+0.05,0.005)
Error= [Errorfunc(j) for j in Beta]

plt.subplot(2,1,1)
plt.plot(Beta,Error,label='mean')
plt.plot(Beta,[2*Errorfunc(j) for j in Beta],label='2*mean')
plt.plot(Beta,[0.2*Errorfunc(j) for j in Beta],label='0.2*mean')
plt.ylabel('Error(MSE)',fontsize=12)
plt.xlabel(r'$\beta$',fontsize=12)
plt.legend()

plt.subplot(2,1,2)
plt.plot(Beta,Error,label='mean')
plt.plot(Beta,[2*Errorfunc(j) for j in Beta],label='2*mean')
plt.plot(Beta,[0.2*Errorfunc(j) for j in Beta],label='0.2*mean')
plt.ylabel('Error(MSE)',fontsize=12)
plt.xlabel(r'$\beta$',fontsize=12)
plt.xlim(2.4,3.6)
plt.ylim(-10,100.3)

plt.savefig('lr1-2.png', dpi=200, bbox_inches="tight")

plt.show()