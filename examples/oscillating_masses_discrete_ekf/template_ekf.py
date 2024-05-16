import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc


def template_ekf(model):
    """
    --------------------------------------------------------------------------
    template_ekf: tuning parameters
    --------------------------------------------------------------------------
    """

    # setting up model variances with a generic value
    q = 1e-5 * np.ones(model.x.shape)
    r = 0.001 * np.ones(model.y.shape)
    Q = np.diag(q.flatten())
    R = np.diag(r.flatten())

    # init
    ekf = do_mpc.estimator.EKF(model=model, Q=Q, R=R)

    # time step
    ekf.settings.t_step = 0.5

    ekf.setup()

    # return
    return ekf