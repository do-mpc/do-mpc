from typing import Any, Dict, Tuple

import numpy as np

from do_mpc.model import Model
from do_mpc.simulator import Simulator
from do_mpc.tools.algebraic_variable_initializer import \
    init_algebraic_variables


def setup_simple_model() -> Model:
    model_type = 'continuous'
    model = Model(model_type)
    x = model.set_variable('_x', 'x')
    z = model.set_variable('_z', 'z')
    z_2 = model.set_variable('_z', 'z_2')
    u = model.set_variable('_u', 'u')
    p = model.set_variable('_p', 'p')

    model.set_rhs('x',  u + p)
    model.set_alg('z_alg_0', z - 1)
    model.set_alg('z_alg_1', z_2 - 1)
    model.setup()
    return model


def create_simple_simulator(model: Model) -> Simulator:
    simulator = Simulator(model)
    simulator.set_param(t_step=1, integration_tool='idas')
    p_template = simulator.get_p_template()

    def p_fun(t_now):
        p_template['p'] = 1
        return p_template

    simulator.set_p_fun(p_fun)
    simulator.x0 = 1
    return simulator


def setup_simple_simulator() -> Simulator:
    model = setup_simple_model()
    simulator = create_simple_simulator(model)
    simulator.setup()
    return simulator


def test_algebraic_variable_initializer() -> None:
    model = setup_simple_model()
    simulator = create_simple_simulator(model)
    init_algebraic_variables(model, simulator)
    np.testing.assert_array_equal(np.array(simulator.z0.master), [[1], [1]])


def test_algebraic_variable_initializer_with_simulator() -> None:
    simulator = setup_simple_simulator()
    simulator.make_step(np.array([[0]]))
    np.testing.assert_array_equal(np.array(simulator.z0.master), [[1], [1]])


if __name__ == '__main__':
    test_algebraic_variable_initializer()
    test_algebraic_variable_initializer_with_simulator()