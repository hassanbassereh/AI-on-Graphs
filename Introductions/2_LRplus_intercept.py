import numpy as np
import matplotlib.pyplot as plt

# Synthetic data
intercept = -4
noise = 7
xdata = np.arange(20)

ydata=[3*i+ intercept +np.random.uniform(-noise,noise) for i in xdata]


def Errorfunc(beta, beta0):
    return np.mean((ydata - (beta * xdata + beta0)) ** 2)

Beta  = np.arange(-8, 15, 0.05)   # x-axis
Beta0 = np.arange(-15, 15, 0.05)   # y-axis

# Z[i, j] -> beta0 = Beta0[i] (row), beta = Beta[j] (column)
Z = np.array([[Errorfunc(b, b0) for b in Beta] for b0 in Beta0])

i, j = np.unravel_index(np.argmin(Z), Z.shape)
print(f"beta  = {Beta[j]}")
print(f"beta0 = {Beta0[i]}")
print(f"min MSE = {Z[i, j]:.4f}")

plt.figure(figsize=(5, 20))

plt.subplot(5, 1, 1)
plt.scatter(xdata, ydata, color='green')
plt.xlabel('x (hours)', fontsize=12)
plt.ylabel('y (scores)', fontsize=12)
plt.ylim(-15,60)
plt.subplot(5, 1, 2)
plt.imshow(Z, cmap='gist_rainbow',origin='lower', aspect='auto',vmax=50,
extent=[Beta.min(), Beta.max(), Beta0.min(), Beta0.max()])
cbar = plt.colorbar()
cbar.set_label('Error (MSE)', fontsize=12)
plt.xlabel(r'$\beta$', fontsize=12)
plt.ylabel(r'$\beta_0$', fontsize=12)



plt.subplot(5, 1, 3)


learningrate = 0.00001
betai = 10
betai0= 7
Path_equal=[]
for q in range(1500):
    if q==0:
        plt.scatter(q,betai,c='r',s=5)
        plt.scatter(q,betai0,c='g',s=5)
    else:
        if q==1:
            plt.scatter(q,betai,c='b',s=5,label=r'$\beta$')
            plt.scatter(q,betai0,c='y',s=5,label=r'$\beta_0$')
        else:
            plt.scatter(q,betai,c='b',s=5)
            plt.scatter(q,betai0,c='y',s=5)

    ypred = [(betai*z)+betai0 for z in xdata]
    yy = [-ypred[i]+ydata[i] for i in range(len(ydata))]
    min2x = [-2*xx for xx in xdata]
    gradient_betai = sum([min2x[i]*yy[i] for i in range(len(yy))])
    gradient_betai0 = -2*sum(yy)
    Path_equal.append((betai, betai0, Errorfunc(betai, betai0)))

    betai = betai - learningrate*gradient_betai
    betai0 = betai0 - learningrate*gradient_betai0
print(betai)

plt.text(400,3,r'$\eta=1*10^{-5}$',fontsize=15,color='b')
plt.text(400,7,r'$\eta=1*10^{-5}$',fontsize=15,color='y')
plt.legend(fontsize=12)
plt.xlabel('epoch',fontsize=12)


plt.subplot(5, 1, 4)

learningrate = 0.00001
betai = 10
betai0= 7
Path_notequal=[]
for q in range(1500):
    if q==0:
        plt.scatter(q,betai,c='r',s=5)
        plt.scatter(q,betai0,c='g',s=5)
    else:
        if q==1:
            plt.scatter(q,betai,c='b',s=5,label=r'$\beta$')
            plt.scatter(q,betai0,c='y',s=5,label=r'$\beta_0$')
        else:
            plt.scatter(q,betai,c='b',s=5)
            plt.scatter(q,betai0,c='y',s=5)

    ypred = [(betai*z)+betai0 for z in xdata]
    yy = [-ypred[i]+ydata[i] for i in range(len(ydata))]
    min2x = [-2*xx for xx in xdata]
    gradient_betai = sum([min2x[i]*yy[i] for i in range(len(yy))])
    gradient_betai0 = -2*sum(yy)
    Path_notequal.append((betai, betai0, Errorfunc(betai, betai0)))

    betai = betai - learningrate*gradient_betai
    betai0 = betai0 - 5e-4*gradient_betai0
print(betai)

plt.text(400,8,r'$\eta=1*10^{-5}$',fontsize=15,color='b')
plt.text(400,-10,r'$\eta=5*10^{-4}$',fontsize=15,color='y')
plt.legend(fontsize=12)
plt.xlabel('epoch',fontsize=12)



plt.subplot(5, 1, 5)

learningrate = 0.00001
betai = 10
betai0= 7
for q in range(1500):
    plt.scatter(q,Errorfunc(betai, betai0),c='purple',s=5)

    ypred = [(betai*z)+betai0 for z in xdata]
    yy = [-ypred[i]+ydata[i] for i in range(len(ydata))]
    min2x = [-2*xx for xx in xdata]
    gradient_betai = sum([min2x[i]*yy[i] for i in range(len(yy))])
    gradient_betai0 = -2*sum(yy)

    betai = betai - learningrate*gradient_betai
    betai0 = betai0 - 5e-4*gradient_betai0
print(betai)


learningrate = 0.00001
betai = 10
betai0= 7
for q in range(1500):
    plt.scatter(q,Errorfunc(betai, betai0),c='g',s=5)

    ypred = [(betai*z)+betai0 for z in xdata]
    yy = [-ypred[i]+ydata[i] for i in range(len(ydata))]
    min2x = [-2*xx for xx in xdata]
    gradient_betai = sum([min2x[i]*yy[i] for i in range(len(yy))])
    gradient_betai0 = -2*sum(yy)

    betai = betai - learningrate*gradient_betai
    betai0 = betai0 - learningrate*gradient_betai0

plt.text(400,100,r'$Same \:\:\:\eta$',fontsize=15,color='g')
plt.text(400,150,r'$Different \:\:\:\eta$',fontsize=15,color='purple')
plt.xlabel('epoch',fontsize=12)
plt.ylabel('Error (EMS)',fontsize=12)
plt.ylim([0,500])




plt.savefig('lr2-1.png', dpi=200, bbox_inches='tight')
plt.close()



import matplotlib as mpl

plt.figure(figsize=(20,10))
B, B0 = np.meshgrid(Beta, Beta0)

ax = plt.subplot(1, 2, 1, projection='3d')
cmap = mpl.colormaps['gist_rainbow'].copy()
surf = ax.plot_surface(B, B0, Z, cmap=cmap,
                       linewidth=0, antialiased=True,
                       rcount=100, ccount=100)
cbar = plt.colorbar(surf, ax=ax, shrink=0.7, extend='max')
cbar.set_label('Error (MSE)', fontsize=12)
ax.set_xlabel(r'$\beta$', fontsize=12)
ax.set_ylabel(r'$\beta_0$', fontsize=12)
ax.set_zlabel('Error (MSE)', fontsize=12)
ax.view_init(elev=40, azim=-105)

ax = plt.subplot(1, 2, 2, projection='3d')
cmap = mpl.colormaps['gist_rainbow'].copy()
cmap.set_over(cmap(1.0))
norm = mpl.colors.Normalize(vmin=Z.min(), vmax=50)   # clip=False is the default
surf = ax.plot_surface(B, B0, Z, cmap=cmap, norm=norm,
                       linewidth=0, antialiased=True,
                       rcount=100, ccount=100)
cbar = plt.colorbar(surf, ax=ax, shrink=0.7, extend='max')
cbar.set_label('Error (MSE)', fontsize=12)
ax.set_xlabel(r'$\beta$', fontsize=12)
ax.set_ylabel(r'$\beta_0$', fontsize=12)
ax.set_zlabel('Error (MSE)', fontsize=12)
ax.view_init(elev=40, azim=-105)
plt.savefig('lr2-2.png', dpi=300, bbox_inches='tight')
plt.close()



Peq  = np.array(Path_equal)
Pneq = np.array(Path_notequal)


import matplotlib as mpl

plt.figure(figsize=(20,10))
B, B0 = np.meshgrid(Beta, Beta0)



ax = plt.subplot(1, 2, 1, projection='3d') 
cmap = mpl.colormaps['gist_rainbow'].copy()
cmap.set_over(cmap(1.0))
norm = mpl.colors.Normalize(vmin=Z.min(), vmax=50)   # clip=False is the default
surf = ax.plot_surface(B, B0, Z, cmap=cmap, norm=norm,
                       linewidth=0, antialiased=True,
                       rcount=100, ccount=100)
ax.scatter(*Peq[0, :3], c='k', s=200, marker='*',
           edgecolors='w', linewidths=1, depthshade=False, zorder=12)
ax.plot(Peq[:, 0], Peq[:, 1], Peq[:, 2],
        color='k', lw=2.5, zorder=10)
cbar = plt.colorbar(surf, ax=ax, shrink=0.7, extend='max')
cbar.set_label('Error (MSE)', fontsize=12)
ax.set_title(r'$Same\:\:\: \eta$', fontsize=22)
ax.set_xlabel(r'$\beta$', fontsize=12)
ax.set_ylabel(r'$\beta_0$', fontsize=12)
ax.set_zlabel('Error (MSE)', fontsize=12)
ax.view_init(elev=40, azim=-105)

ax = plt.subplot(1, 2, 2, projection='3d') 
cmap = mpl.colormaps['gist_rainbow'].copy()
cmap.set_over(cmap(1.0))
norm = mpl.colors.Normalize(vmin=Z.min(), vmax=50)   # clip=False is the default
surf = ax.plot_surface(B, B0, Z, cmap=cmap, norm=norm,
                       linewidth=0, antialiased=True,
                       rcount=100, ccount=100)
ax.scatter(*Pneq[0, :3], c='k', s=200, marker='*',
           edgecolors='w', linewidths=1, depthshade=False, zorder=12)
ax.plot(Pneq[:, 0], Pneq[:, 1], Pneq[:, 2],
        color='k', lw=2.5, zorder=10)
cbar = plt.colorbar(surf, ax=ax, shrink=0.7, extend='max')
cbar.set_label('Error (MSE)', fontsize=12)
ax.set_title(r'$Different\:\:\: \eta$', fontsize=22)

ax.set_xlabel(r'$\beta$', fontsize=12)
ax.set_ylabel(r'$\beta_0$', fontsize=12)
ax.set_zlabel('Error (MSE)', fontsize=12)
ax.view_init(elev=40, azim=-105)
plt.savefig('lr2-3.png', dpi=300, bbox_inches='tight')
plt.show()

