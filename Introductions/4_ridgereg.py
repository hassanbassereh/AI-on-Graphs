import numpy as np, matplotlib.pyplot as plt
def orth(v,*c):
    A=np.column_stack([np.ones(len(v)),*c]); return v-A@np.linalg.lstsq(A,v,rcond=None)[0]

np.random.seed(0)
N=20; noise=7
x1data=np.arange(N,dtype=float)
x2data=orth(np.random.normal(0,10,N), x1data)
x3data=5*x1data+np.random.normal(0,1.5,N)
x4data=orth(np.random.normal(0,10,N), x1data)
ydata=2.1*x1data+np.random.uniform(-noise,noise,N)

X=np.column_stack([np.ones(N),x1data,x2data,x3data,x4data]); y=ydata
XtX=X.T@X; Xty=X.T@y
P=np.eye(5); P[0,0]=0

# ================= CROSS-VALIDATION =================
def ridge_fit(Xm, ym, lam):
    return np.linalg.solve(Xm.T@Xm + lam*P, Xm.T@ym)

def cv_error(lam, n_rep=40, k=5):
    tot = 0.0
    for rep in range(n_rep):
        idx = np.random.default_rng(rep).permutation(N)
        for f in np.array_split(idx, k):
            m = np.ones(N, bool); m[f] = False
            b = ridge_fit(X[m], y[m], lam)
            tot += np.sum((y[f] - X[f] @ b)**2)
    return tot/(N*n_rep)

lams = np.logspace(-3, 4, 120)
cv   = np.array([cv_error(l) for l in lams])
lam_opt = lams[cv.argmin()]

beta_cv = np.linalg.solve(XtX + lam_opt*P, Xty)
print(f'lambda_opt = {lam_opt:.3f}   (CV MSE {cv.min():.2f} vs {cv[0]:.2f} at lam=0)')
print('beta OLS      ', np.round(np.linalg.solve(XtX,Xty)[1:],4))
print('beta ridge CV ', np.round(beta_cv[1:],4))

# ================= PLOT =================
fig, ax = plt.subplots(1,2, figsize=(13,5))
ax[0].plot(lams, cv, 'k', lw=2.5)
ax[0].axvline(lam_opt, color='b', ls=':', lw=2, label=fr'$\lambda_{{opt}}$={lam_opt:.2f}')
ax[0].scatter([lam_opt],[cv.min()], color='b', s=90, zorder=5)
ax[0].scatter([lams[0]],[cv[0]], color='orange', s=90, zorder=5)
ax[0].annotate('OLS',(lams[0],cv[0]),textcoords='offset points',xytext=(10,8),
               color='orange',fontsize=12)
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel(r'$\lambda$',fontsize=13); ax[0].set_ylabel('CV error',fontsize=13)
ax[0].set_title('5-fold cross-validation',fontsize=13); ax[0].legend(fontsize=11)

paths=np.array([np.linalg.solve(XtX+l*P,Xty) for l in lams])
for i,(nm,c) in enumerate(zip([r'$b_1$',r'$b_2$',r'$b_3$',r'$b_4$'],
                              ['b','g','r','gray']),start=1):
    ax[1].plot(lams,paths[:,i],color=c,lw=2,label=nm)
ax[1].axvline(lam_opt,color='k',ls=':',lw=2); ax[1].axhline(0,color='k',lw=.7)
ax[1].set_xscale('log'); ax[1].set_xlabel(r'$\lambda$',fontsize=13)
ax[1].set_ylabel('coefficient',fontsize=13)
ax[1].set_title(r'Coefficient path',fontsize=13); ax[1].legend(fontsize=11)
plt.tight_layout()
plt.savefig('4_cv_plot.png', dpi=200, bbox_inches='tight')
plt.close()

exit()


import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

# ------------------------------------------------------------------
# Model:  y = beta * x + noise      (single parameter, as before)
# Regime where ridge actually helps: few points, short x-range,
# heavy noise -> the OLS estimate is very unstable.
# ------------------------------------------------------------------
beta_true = 3.0
n         = 10
sigma     = 6.0
xdata     = np.linspace(0, 5, n)
S         = np.sum(xdata**2)

# Cost:  J(b) = mean((y - b*x)^2) + lam * b^2
# dJ/db = 0  ->  b = sum(x*y) / (sum(x^2) + n*lam)
def beta_hat(y, lam):
    return (y @ xdata) / (S + n * lam)

def cost(b, y, lam):
    return np.mean((y - b * xdata) ** 2) + lam * b ** 2

# Shrinkage factor c(lam) = b_ridge / b_ols
def shrink(lam):
    return S / (S + n * lam)

# bias^2 = (c-1)^2 * beta^2 ,  var = c^2 * sigma^2 / S
# minimising the sum gives:
lam_star = sigma ** 2 / (n * beta_true ** 2)

# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# --- Panel 1: one sample, OLS vs ridge ----------------------------
ax = axes[0, 0]
y_one = beta_true * xdata + rng.normal(0, sigma, n)
b_ols, b_rid = beta_hat(y_one, 0), beta_hat(y_one, lam_star)

ax.scatter(xdata, y_one, color='k', zorder=3, label='data')
xx = np.linspace(0, 2, 50)
ax.plot(xx, beta_true * xx, 'k--', lw=2, label=fr'truth  $\beta$={beta_true:.1f}')
ax.plot(xx, b_ols * xx, 'orange', lw=2, label=fr'OLS  $\hat\beta$={b_ols:.2f}')
ax.plot(xx, b_rid * xx, 'b', lw=2, label=fr'ridge  $\hat\beta$={b_rid:.2f}')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('One noisy sample', fontsize=13)
ax.legend(fontsize=9)

# --- Panel 2: sampling distribution over many datasets ------------
ax = axes[0, 1]
n_rep = 4000
Y = beta_true * xdata + rng.normal(0, sigma, size=(n_rep, n))
B_ols = Y @ xdata / S
B_rid = Y @ xdata / (S + n * lam_star)

bins = np.linspace(-6, 12, 70)
ax.hist(B_ols, bins=bins, color='orange', alpha=.65,
        label=fr'OLS   sd={B_ols.std():.2f}')
ax.hist(B_rid, bins=bins, color='blue', alpha=.55,
        label=fr'ridge sd={B_rid.std():.2f}')
ax.axvline(beta_true, color='k', ls='--', lw=2, label=r'true $\beta$')
ax.set_xlabel(r'$\hat\beta$', fontsize=12)
ax.set_ylabel('count', fontsize=12)
ax.set_title(f'4000 repeated datasets  ($\\lambda$={lam_star:.2f})', fontsize=13)
ax.legend(fontsize=9)

# --- Panel 3: bias-variance decomposition vs lambda ---------------
ax = axes[1, 0]
lams = np.linspace(0, 3, 400)
c = shrink(lams)
bias2 = (c - 1) ** 2 * beta_true ** 2
var = c ** 2 * sigma ** 2 / S

ax.plot(lams, bias2, 'r', lw=2, label=r'bias$^2$')
ax.plot(lams, var, 'g', lw=2, label='variance')
ax.plot(lams, bias2 + var, 'k', lw=2.5, label=r'total error')
ax.axvline(lam_star, color='b', ls=':', lw=2,
           label=fr'$\lambda^*$={lam_star:.2f}')
ax.scatter([0], [var[0]], color='orange', s=70, zorder=5)
ax.annotate('OLS', (0, var[0]), textcoords='offset points',
            xytext=(12, 6), color='orange', fontsize=11)
ax.set_xlabel(r'$\lambda$', fontsize=12)
ax.set_ylabel(r'error in $\hat\beta$', fontsize=12)
ax.set_title('Why a penalty helps', fontsize=13)
ax.legend(fontsize=9)

# --- Panel 4: the cost function being minimised -------------------
ax = axes[1, 1]
bgrid = np.linspace(-2, 9, 400)
for lam, col in [(0, 'orange'), (lam_star, 'blue'), (5 * lam_star, 'red')]:
    J = [cost(b, y_one, lam) for b in bgrid]
    ax.plot(bgrid, J, color=col, lw=2, label=fr'$\lambda$={lam:.2f}')
    bm = beta_hat(y_one, lam)
    ax.axvline(bm, color=col, ls=':', lw=1.2)
ax.axvline(beta_true, color='k', ls='--', lw=2, label=r'true $\beta$')
ax.set_xlabel(r'$\beta$', fontsize=12)
ax.set_ylabel(r'$J(\beta)$', fontsize=12)
ax.set_title(r'Penalty pulls the minimum toward 0', fontsize=13)
ax.set_ylim(0, 300)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('4_ridge.png', dpi=200, bbox_inches='tight')

# ------------------------------------------------------------------
print(f'lambda* = {lam_star:.3f}')
print(f'OLS   : bias={B_ols.mean()-beta_true:+.3f}  sd={B_ols.std():.3f}  '
      f'MSE={np.mean((B_ols-beta_true)**2):.3f}')
print(f'ridge : bias={B_rid.mean()-beta_true:+.3f}  sd={B_rid.std():.3f}  '
      f'MSE={np.mean((B_rid-beta_true)**2):.3f}')