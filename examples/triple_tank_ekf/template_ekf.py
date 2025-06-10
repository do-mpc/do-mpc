import numpy as np
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

def template_ekf(model):
    # init
    ekf = do_mpc.estimator.EKF(model=model)

    # modifying ekf settings
    ekf.settings.t_step = 1

    p_template = ekf.get_p_template()
    # Typically, the values would be reset at each call of p_fun.
    # Here we just return the fixed values:
    def p_fun_ekf(t_now):
        p_template['p1'] = 2
        return p_template
    ekf.set_p_fun(p_fun_ekf)


    tvp_template = ekf.get_tvp_template()
    # Typically, the values would be reset at each call of tvp_fun.
    # Here we just return the fixed values:
    def tvp_fun_ekf(t_now):
        if t_now<50:
            tvp_template['tvp1'] = 0.5
        else:
            tvp_template['tvp1'] = 1
        return tvp_template
    ekf.set_tvp_fun(tvp_fun_ekf)

    ekf.setup()

    # return the estimator
    return ekf