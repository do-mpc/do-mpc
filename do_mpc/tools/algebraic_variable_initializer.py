"""Module to initialize the algebraic variables."""
from typing import Any

import casadi as cd

from do_mpc.model import Model

SUPPRESS_IPOPT = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}


def init_algebraic_variables(
    model: Model,
    obj: Any,
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
    algebraic_equations = cd.vertcat(model._alg)
    variables_to_substitute = cd.vertcat(model.p, model.tvp, model.u, model.x)
    initial_guess = cd.vertcat(obj.p_fun(obj.t0),
                               obj.tvp_fun(obj.t0), obj.u0,
                               obj.x0)
    initialized_algebraic_equations = cd.substitute(algebraic_equations,
                                                    variables_to_substitute,
                                                    initial_guess)
    nlp['g'] = initialized_algebraic_equations

    solver = cd.nlpsol("solver", "ipopt", nlp, SUPPRESS_IPOPT)
    res = solver(x0=z0, lbg=0, ubg=0)
    z_init = res['x']
    obj.z0 = z_init