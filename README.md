# sfvp

Python code for computing stream function and velocity potential on a regular grid 
(constant dx, constant dy), given the x and y components of a wind field. 
The code isn't very fast but it should work on all platforms. If you're 
on a linux or mac, you're probably better off using something like 
[windspharm](https://ajdawson.github.io/windspharm/latest/) instead. 

## Methods

### Sources
* The implementation follows: Li, Zhijin and Chao, Yi and McWilliams, James C., 2006: 
Computation of the Streamfunction and Velocity Potential for Limited and Irregular Domains. 
Monthly Weather Review, 3384-3394.
* This [repository](https://github.com/Xunius/py_helmholtz) was also very helpful. 

### Summary
The code calculates the stream function and velocity potential
from an arbitrary velocity field. This is done through a process known as 
[Helmholtz Decomposition](https://en.wikipedia.org/wiki/Helmholtz_decomposition). 

```
(u, v) = = grad(chi) + cross(k-hat, grad(psi)) = (u_irrot, v_irrot) + (u_nondiv, v_nondiv) 

psi = stream function
chi = velocity potential
u_irrot = u_chi = dchi / dx
v_irrot = v_chi = dchi / dy
u_nondiv = u_psi = -dpsi / dy
v_nondiv = v_psi = dpsi / dx
```

The stream function (`psi`) and velocity potential (`chi`) are approximated iteratively. 

Initial guess at `psi` and `chi`: Can be calculated a number of ways, but drawing 
randomly from a normal distribution is most straightforward. 

Cost function: 
```
J = SUM((uhat - u)**2 + (vhat - v)**2) + lambda * SUM(chi_hat**2 + psi_hat*2)

uhat = uhat_psi + uhat_chi = -dpsi_hat / dy + chi_hat / dx
vhat = vhat_psi + vhat_chi = dpsi_hat / dx + chi_hat / dy
chi_hat = predicted velocity potential field
psi_hat = predicted stream function field
```

The partial derivative of cost the cost function with respect to each parameter can 
be split into two parts. the first is related to the regularization term 
(`lambda * SUM(chi_hat**2 + psi_hat*2)`):
```
dJ_reg / d_param = 2 * lambda * param

param = any arbitrary value in the psi or chi field.
```

The second part is related to the local partial derivatives of psi and chi. 
This is more complicated. 

```
dJ / dchi = -d(err_u) / dx - d(err_v) / dy
dJ / dpsi = -d(err_v) / dx + d(err_u) / dy 

err_u = (uhat - u)
err_v = (vhat - v)
```

Scipy.optimize.minimize is then used to solve for psi and chi, given u and v. 
The calculations occur on a staggered grid, where psi and chi are defined at the center 
of grid boxes defined by u/v. Note that the psi/chi grid is extended outside 
the u/v grid. 

The code does not make any assumptions about boundary conditions. Allowing for 
periodic boundary conditions in x or y would probably be straightforward with 
some modifications to `shift2d` (see code). 

## Requirements

Python 3, numpy, scipy

## Example usage

```python
import numpy as np
from sfvp import sfvp

x = np.linspace(0, 360, 181)
y = np.linspace(0, 180, 91)

X, Y = np.meshgrid(x, y)

u = (np.sin(X / 20) - np.sin(Y / 20)) * 10
v = (np.cos(X / 20 + 5) + np.sin(Y / 20 - 5)) * 10

sf, vp = sfvp(u, v, x, y, lam=1e-8, guess_method='randn', interp=True)
```

