# %%
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

import os 
import sys

sys.path.append(os.path.join('..','..'))
import do_mpc
# %%

def get_optim():
    x = ca.SX.sym('x', 2)
    p = ca.SX.sym('p', 1)

    cost = (1-x[0])**2 + 0.2*(x[1]-x[0]**2)**2

    cons_inner = (x[0] + 0.5)**2+x[1]**2

    cons = ca.vertcat(
        p**2/4 - cons_inner,
        cons_inner - p**2
    )
    
    nlp = {'x':x, 'p':p, 'f':cost, 'g':cons}
    nlp_bounds = {
        'lbx': np.array([0, -ca.inf]).reshape(-1,1), 
        'ubx':np.array([ca.inf, ca.inf]).reshape(-1,1), 
        'lbg':np.array([-ca.inf, -ca.inf]).reshape(-1,1), 
        'ubg':np.array([0, 0]).reshape(-1,1)
        }

    solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level':0 , 'ipopt.sb': 'yes', 'print_time':0, 'ipopt.tol': 1e-14})

    return nlp, nlp_bounds, solver

def get_optim_diff(nlp, nlp_bounds):
    # instantiate NLPDifferentiator
    nlp_diff = do_mpc.differentiator.NLPDifferentiator(nlp, nlp_bounds)
    nlp_diff.settings.check_LICQ = True
    nlp_diff.settings.check_rank = True
    nlp_diff.settings.track_residuals = True

    return nlp_diff


nlp, nlp_bounds, solver = get_optim()
nlp_diff = get_optim_diff(nlp, nlp_bounds)


# %%
p_test = np.linspace(0, 2.5, 50)
x_test = np.zeros((len(p_test), 2))
dxdp_test = np.zeros((len(p_test), 2))
for i,p_i in enumerate(p_test):
    print(i, end='\r')
    r = solver(p=p_i, **nlp_bounds)
    x_test[i] = r['x'].full().flatten()

    dxdp, _ = nlp_diff.differentiate(r, p_i)
    dxdp_test[i] = dxdp.full().flatten()
# %%
fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(p_test, x_test[:,0], label='$x_0^*(p)$')
ax[1].plot(p_test, x_test[:,1], label='$x_1^*(p)$')

ax[0].plot(p_test, dxdp_test[:,0], label='$\partial_p x_0^*(p)$')
ax[1].plot(p_test, dxdp_test[:,1], label='$\partial_p x_1^*(p)$')

ax[0].quiver(p_test, x_test[:,0], np.ones_like(p_test), dxdp_test[:,0], angles='xy')
ax[1].quiver(p_test, x_test[:,1], np.ones_like(p_test), dxdp_test[:,1], angles='xy')

ax[1].set_xlabel('$p$')

ax[0].legend()
ax[1].legend()

ax[0].set_title('Optimal solution and sensitivity depending on parameter $p$')

plt.show(block=True)
# %%
