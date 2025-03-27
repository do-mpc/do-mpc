import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc


def template_simulator(model):
    """
    --------------------------------------------------------------------------
    template_optimizer: tuning parameters
    --------------------------------------------------------------------------
    """
    simulator = do_mpc.simulator.Simulator(model)


    simulator.set_param(t_step = 1)

    p_template = simulator.get_p_template()
    def p_fun(t_now):
        p_template['p1'] = 2
        return p_template
    simulator.set_p_fun(p_fun)

    # The timevarying paramters have no effect on the simulator (they are only part of the cost function).
    # We simply use the default values:
    tvp_template = simulator.get_tvp_template()
    def tvp_fun(t_now):
        if t_now<50:
            tvp_template['tvp1'] = 0.5
        else:
            tvp_template['tvp1'] = 1
        return tvp_template

    simulator.set_tvp_fun(tvp_fun)

    simulator.setup()

    return simulator
