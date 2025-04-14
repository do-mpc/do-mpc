import numpy as np
import do_mpc
from casadi import DM
import math


def system_model(delta_t, omega):
    model_type = 'discrete'  # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)
    print('omega: ', omega)
    print('e^omega dt: ', np.exp(omega * delta_t))
    b = model.set_variable(var_type='_x', var_name='b', shape=(1, 1))
    u = model.set_variable(var_type='_u', var_name='u', shape=(1, 1))
    # Define disturbance as a parameter (e.g., d)
    d = model.set_variable(var_type='_p', var_name='d', shape=(1, 1))
    b_next = b * np.exp(omega * delta_t) - u + d
    model.set_rhs('b', b_next)
    model.setup()
    return model


def create_simulator(model, delta_t, omega_n):
    simulator = do_mpc.simulator.Simulator(model)
    # Setup the simulator
    params_simulator = {
        't_step': delta_t  # Set the time step for simulation
    }
    simulator.set_param(**params_simulator)

    p_num = simulator.get_p_template()
    p_num['d'] = 0.0

    def disturbance_function(t_ind):
        """
        This function defines how the disturbance parameter 'd' evolves over time.
        t_ind is the time index.
        Surface acceleration: \ddot{x}_s = A*sin(2pi/t_e*t)
        t_initial = t_ind
        t_end = t_ind + delta_t
        This function returns: - 1/omega integration from t_initial to t_end (exp(omega (t_end - tau)) \ddot{x}_s) dtau
        """
        a_external = 0.1  # The magnitude of acceleration disturbance
        t_external = 0.5  # The period of acceleration disturbance
        omega_external = 2 * math.pi / t_external

        t_initial = t_ind
        t_end = t_ind + delta_t

        term_1 = omega_n * math.sin(omega_external * t_initial) + omega_external * math.cos(omega_external * t_initial)
        term_2 = omega_n * math.sin(omega_external * t_end) + omega_external * math.cos(omega_external * t_end)
        numerator = math.exp(omega_n * delta_t) * term_1 - term_2
        denominator = omega_external ** 2 + omega_n ** 2
        p_num['d'] = -1 / omega_n * a_external * numerator / denominator
        print('disturbance: ', p_num['d'])
        return p_num

    # Set the disturbance parameter function
    simulator.set_p_fun(disturbance_function)
    simulator.setup()
    return simulator


def terminal_controller(delta_t, omega, b_measure):
    """
    b_desire = b_desire * exp(omega*delta_t) - u_desire
    """
    u_desire = 0.03
    b_desire = u_desire / (math.exp(omega * delta_t) - 1)
    print('b_desire: ', b_desire)
    u = u_desire + (b_measure - b_desire) * math.exp(omega * delta_t)
    u_max = 0.1
    u_min = -0.1
    u_saturated = min(u_max, max(u_min, u))
    print('u_saturated: ', u_saturated)
    return u_saturated


def run_simulation():
    delta_t = 0.2
    g = 9.8
    com_height = 0.55
    omega = np.sqrt(g / com_height)
    model = system_model(delta_t, omega)
    sim = create_simulator(model, delta_t, omega)
    b = 0.0603  # initial value
    sim.x0 = np.array([[b]])
    num_steps = 50

    states = []
    inputs = []

    for i in range(num_steps):
        states.append(b)
        u = terminal_controller(delta_t, omega, b_measure=b)
        inputs.append(u)
        b = sim.make_step(np.array([u]).reshape(1, 1))
    print('states: ', states)
    print('inputs: ', inputs)

run_simulation()
