# imports
from casadi import *
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc


def template_model(symvar_type = 'SX'):

    # model type
    model_type = 'discrete'

    # initialisation of model
    model = do_mpc.model.Model(model_type=model_type, symvar_type=symvar_type)

    # states
    x1 = model.set_variable(var_type='_x', var_name='x1')
    x2 = model.set_variable(var_type='_x', var_name='x2')
    x3 = model.set_variable(var_type='_x', var_name='x3')


    # inputs
    u1 = model.set_variable(var_type='_u', var_name='u1')
    u2 = model.set_variable(var_type='_u', var_name='u2')

    # State measurements
    model.set_meas('x3_meas', x3)

    p1 = model.set_variable(var_type='_p', var_name='p1')
    tvp1 = model.set_variable(var_type='_tvp', var_name='tvp1')
    # defining auxiliary variables
    A = 0.00154
    g = 9.81
    Ts = 1
    r1 = 1
    r2 = 0.8
    r3 = 1
    sp = 5 * 1e-5
    q13 = r1 * sp * sign(x1 - x3) * sqrt(2 * g * fabs(x1 - x3))
    q32 = r3 * sp * sign(x3 - x2) * sqrt(2 * g * fabs(x3 - x2))
    q20 = r2 * sp * sqrt(2 * g * x2) * tvp1 * p1
    #q20 = r2 * sp * sqrt(2 * g * x2)

    # model equations
    x1_next = x1 + (Ts / A) * (-q13 + u1)
    x2_next = x2 + (Ts / A) * (q32 - q20 + u2)
    x3_next = x3 + (Ts / A) * (q13 - q32)

    # saving equations
    model.set_rhs('x1', x1_next)
    model.set_rhs('x2', x2_next)
    model.set_rhs('x3', x3_next)

    # setup
    model.setup()

    # returns the model
    return model