"""Module to initialize the algebraic variables."""
import casadi as cd

from do_mpc.model import IteratedVariables, Model

SUPPRESS_IPOPT = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}


def init_algebraic_variables(
    model: Model,
    obj: IteratedVariables,
) -> None:
    """Initializes the algebraic variables.

    :param model: Dompc model object.
    :type model: Model
    :param obj: Object containing an initial guess for z0.
    :type obj: IteratedVariables
    :raises RuntimeError: Model was not setup.
    """
    if model.flags['setup'] is False:
        raise RuntimeError(
            'The model must be setup before the algebraic variables can be initialized.'
        )
    nlp = {}
    nlp['x'] = cd.vertcat(model.z)
    z0 = cd.vertcat(obj.z0)
    residual_to_initial_guess = cd.vertcat(model.z) - z0
    nlp['f'] = cd.sum2(cd.sum1(residual_to_initial_guess**2))
    nlp['g'] = cd.vertcat(model._alg)

    solver = cd.nlpsol("solver", "ipopt", nlp, SUPPRESS_IPOPT)
    res = solver(x0=z0, lbg=0, ubg=0)
    z_init = res['x']
    obj.z0 = z_init