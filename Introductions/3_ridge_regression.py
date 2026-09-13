import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Synthetic data preparation
def orth(v, *cols):
    """Remove v's component in span{1, cols...} -> correlation exactly 0."""
    A = np.column_stack([np.ones(len(v)), *cols])
    return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]

def unit(v):
    """Normalize to mean 0 and unit norm."""
    v = v - v.mean()
    return v / np.linalg.norm(v)

# Set random seed for reproducibility
np.random.seed(42)

N = 20
noise = 7
x1data = np.arange(N, dtype=float)
ydata = np.array([2.1*i + np.random.uniform(-noise, noise) for i in x1data])
u1 = unit(x1data)
ur = unit(orth(ydata, x1data))  # the part of y that x1 misses
rho = 0.01  # just under the significance threshold
x2data = -10*(rho*u1 + np.sqrt(1-rho**2)*ur)

x3data = 25*x1data + np.random.normal(0, 2.3, N)
x4data = orth(np.random.normal(0, 10, N), x1data, ydata)

# Correlation analysis
X = np.array([x1data, x2data, x3data, x4data])  # rows = variables
C = np.corrcoef(X)
print("Correlation Matrix:")
print(np.round(C, 3))
print()

# Pearson correlations with y
names = ['x1', 'x2', 'x3', 'x4']
print("Correlations with y:")
for name, xs in zip(names, [x1data, x2data, x3data, x4data]):
    r, p = pearsonr(xs, ydata)
    print(f'{name} vs y:  r = {r:+.3f}   p = {p:.3e}')
print()

# Correlation matrix table
print(f"{'':6}" + "".join(f"{n:>18}" for n in names))
cols = [x1data, x2data, x3data, x4data]
for ni, vi in zip(names, cols):
    row = f"{ni:6}"
    for vj in cols:
        r, p = pearsonr(vi, vj)
        row += f"   r={r:+.3f} p={p:.1e}"
    print(row)
print()

# Single variable regressions (beta_hat)
print("Single variable regression coefficients:")
betas_single = []
for nm, x in zip(names, cols):
    beta = np.sum(x * ydata) / np.sum(x**2)
    betas_single.append(beta)
    print(f'{nm}: beta_hat = {beta:+.4f}')
print()

# OLS regression
X_full = np.column_stack([np.ones(N), x1data, x2data, x3data, x4data])  # (20, 5)
y = ydata

XtX = X_full.T @ X_full  # (5, 5)
Xty = X_full.T @ y  # (5,)
beta_ols = np.linalg.solve(XtX, Xty)  # [b0, b1, b2, b3, b4]
print('OLS coefficients (excluding intercept):', np.round(beta_ols[1:], 4))
print()

# Ridge regression
lam = 0.05
P = np.eye(5)
P[0, 0] = 0  # don't shrink b0
beta_ridge = np.linalg.solve(XtX + lam * P, Xty)
print('Ridge coefficients (lambda=0.05, excluding intercept):', np.round(beta_ridge[1:], 4))
print()

# Plotting
fig = plt.figure(figsize=(22, 10))

# Plot 1: Correlation heatmap (FIRST)
ax_heatmap = plt.subplot(2, 5, 1)
all_vars = np.column_stack([x1data, x2data, x3data, x4data])
corr_with_y = np.corrcoef(all_vars.T)
var_names = ['x1', 'x2', 'x3', 'x4']

sns.heatmap(corr_with_y, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            xticklabels=var_names, yticklabels=var_names, ax=ax_heatmap,
            vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
ax_heatmap.set_title('Correlation Heatmap', fontsize=12, fontweight='bold')

# Plot 2-5: Scatter plots with fitted lines (POSITIONS 2-5)
for idx, (name, x) in enumerate(zip(names, cols), 1):
    ax = plt.subplot(2, 5, idx+1)
    ax.scatter(x, ydata, color='green', s=80, alpha=0.6, edgecolors='darkgreen', linewidth=1.5)
    
    # Fitted line from single variable regression
    x_line = np.array([x.min(), x.max()])
    y_line = betas_single[idx-1] * x_line
    if idx==2 or idx==4:
        ax.plot(x_line, y_line+20, 'g-', linewidth=2.5, label=f'β={betas_single[idx-1]:.4f}')
        ax.plot(x_line, beta_ols[idx]* x_line+19.8, 'b--', linewidth=2.5, label=f'β_OLS={beta_ols[idx]:.4f}')
    elif idx==3:
        ax.plot(x_line, y_line, 'g-', linewidth=2.5, label=f'β={betas_single[idx-1]:.4f}')
        ax.plot(x_line, beta_ols[idx]* x_line+19.8, 'b--', linewidth=2.5, label=f'β_OLS={beta_ols[idx]:.4f}')

    else:
        ax.plot(x_line, y_line, 'g-', linewidth=2.5, label=f'β={betas_single[idx-1]:.4f}')
        ax.plot(x_line, beta_ols[idx]* x_line, 'b--', linewidth=2.5, label=f'β_OLS={beta_ols[idx]:.4f}')

  
    ax.set_xlabel(name, fontsize=12, fontweight='bold')
    ax.set_ylabel('y', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lr3-ridge-regression-analysis.png', dpi=200, bbox_inches="tight")
print("Plot saved as 'lr3-ridge-regression-analysis.png'")
plt.close()





# Plotting
fig = plt.figure(figsize=(22, 10))

# Plot 1: Correlation heatmap (FIRST)
ax_heatmap = plt.subplot(2, 5, 1)
all_vars = np.column_stack([x1data, x2data, x3data, x4data])
corr_with_y = np.corrcoef(all_vars.T)
var_names = ['x1', 'x2', 'x3', 'x4']

sns.heatmap(corr_with_y, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            xticklabels=var_names, yticklabels=var_names, ax=ax_heatmap,
            vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
ax_heatmap.set_title('Correlation Heatmap', fontsize=12, fontweight='bold')

# Plot 2-5: Scatter plots with fitted lines (POSITIONS 2-5)
for idx, (name, x) in enumerate(zip(names, cols), 1):
    ax = plt.subplot(2, 5, idx+1)
    ax.scatter(x, ydata, color='green', s=80, alpha=0.6, edgecolors='darkgreen', linewidth=1.5)
    
    # Fitted line from single variable regression
    x_line = np.array([x.min(), x.max()])
    y_line = betas_single[idx-1] * x_line
    if idx==2 or idx==4:
        ax.plot(x_line, y_line+20, 'g-', linewidth=2.5, label=f'β={betas_single[idx-1]:.4f}')
        ax.plot(x_line, beta_ols[idx]* x_line+19.8, 'b--', linewidth=2.5, label=f'β_OLS={beta_ols[idx]:.4f}')
        ax.plot(x_line, beta_ridge[idx]* x_line+19.7, 'r.-', linewidth=2.5, label=f'β_ridge={beta_ridge[idx]:.4f}')
    elif idx==3:
        ax.plot(x_line, y_line, 'g-', linewidth=2.5, label=f'β={betas_single[idx-1]:.4f}')
        ax.plot(x_line, beta_ols[idx]* x_line+19.8, 'b--', linewidth=2.5, label=f'β_OLS={beta_ols[idx]:.4f}')
        ax.plot(x_line, beta_ridge[idx]* x_line+15, 'r.-', linewidth=2.5, label=f'β_ridge={beta_ridge[idx]:.4f}')

    else:
        ax.plot(x_line, y_line, 'g-', linewidth=2.5, label=f'β={betas_single[idx-1]:.4f}')
        ax.plot(x_line, beta_ols[idx]* x_line, 'b--', linewidth=2.5, label=f'β_OLS={beta_ols[idx]:.4f}')
        ax.plot(x_line, beta_ridge[idx]* x_line, 'r.-', linewidth=2.5, label=f'β_ridge={beta_ridge[idx]:.4f}')

  
    ax.set_xlabel(name, fontsize=12, fontweight='bold')
    ax.set_ylabel('y', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lr3-ridge-regression-analysis__fffff.png', dpi=200, bbox_inches="tight")
print("Plot saved as 'lr3-ridge-regression-analysis.png'")
plt.close()