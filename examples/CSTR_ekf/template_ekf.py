import numpy as np
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

def template_ekf(model):
    # init
    ekf = do_mpc.estimator.EKF(model=model)

    #ekf.settings.n_horizon =  10
    ekf.settings.t_step = 0.005

    tvp_num = ekf.get_tvp_template()
    def tvp_fun(t_now):
        return tvp_num

    ekf.set_tvp_fun(tvp_fun)

    p_num = ekf.get_p_template()
    p_num['alpha'] = 1
    p_num['beta'] = 1
    def p_fun(t_now):
        return p_num

    ekf.set_p_fun(p_fun)

    ekf.setup()

    # return the estimator
    return ekf