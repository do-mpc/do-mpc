#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from ._base import Estimator
import do_mpc
from typing import Union
import casadi as ca
import casadi.tools as castools
from typing import Callable
from ..simulator import Simulator
from ._estimatorsettings import EstimatorSettings
from ..model import Model



class EKF(Estimator):
    """Extended Kalman Filter. Setup this class and use :py:func:`EKF.make_step`
    during runtime to obtain the currently estimated states given the measurements ``y0``.

    Warnings:
        Work in progress.
    """
    def __init__(self, model:Union[do_mpc.model.Model,do_mpc.model.LinearModel]):
        
        # init
        Estimator.__init__(self, model)
        self.settings = EstimatorSettings()
        
        # Stores the variances of state(s) and measurement(s)
        # TODO: make Q and R changable by tvp (AT MAKE_STEP)
        #self.Q = Q
        #self.R = R

        # generating flags
        self.flags = {
            'setup': False,
            'set_initial_guess': False,
            'set_tvp_fun': False,
            'set_p_fun': False,
            'first_step': True
        }

    def _check_validity(self):
        # tvp_fun must be set, if tvp are defined in model.
        if self.flags['set_tvp_fun'] == False and self.model._tvp.size > 0:
            raise Exception('You have not supplied a function to obtain the time-varying parameters defined in model. Use .set_tvp_fun() prior to setup.')
        # p_fun must be set, if p are defined in model.
        if self.flags['set_p_fun'] == False and self.model._p.size > 0:
            raise Exception('You have not supplied a function to obtain the parameters defined in model. Use .set_p_fun() prior to setup.')

        # Set dummy functions for tvp and p in case these parameters are unused.
        if not self.flags['set_tvp_fun']:
            _tvp = self.get_tvp_template()
            def tvp_fun(t): return _tvp
            self.set_tvp_fun(tvp_fun)

        if not self.flags['set_p_fun']:
            _p = self.get_p_template()
            def p_fun(t): return _p
            self.set_p_fun(p_fun)
    
        return None
    
    def setup(self):

        assert self.model._alg.shape[0] == 0, 'EKF with Algebraic states not ready for use.'

        # extracting model information
        x, u, z, tvp, p, w = self.model['x', 'u', 'z', 'tvp', 'p', 'w']

        # extracting linearised system matrices
        A, B, C, D = self.model.get_linear_system_matrices()

        # generating functions
        self.A_fun = ca.Function('A_fun', [x, u, p, tvp], [A])
        self.B_fun = ca.Function('B_fun', [x, u, p, tvp], [B])
        self.C_fun = ca.Function('C_fun', [x, u, p, tvp], [C])
        self.D_fun = ca.Function('D_fun', [x, u, p, tvp], [D])

        # algebraic equation
        alg = self.model._alg

        # setting up integrator for model.model_type = 'continuous'
        if self.model.model_type == 'continuous':

            # state equations
            f = self.model._rhs

            # measurement equation
            h = self.model._y_expression
            self.h_fun = ca.Function('h_fun', [x, u, p, tvp, self.model._v], [h])

            # storing time step
            opts = {'tf': self.settings.t_step}

            # init covariance matrix (symbolic)
            P = ca.SX.sym('P', self.model.n_x, self.model.n_x)
            Q = ca.SX.sym('Q', self.model.n_x, self.model.n_x)

            # covariance integration equation
            dP_dt = (A @ P) + (P @ A.T) + Q

            # setting up dict
            #combined_ode = {'x': ca.vertcat(x, P.reshape((-1, 1))),
            #                'ode': ca.vertcat(f, dP_dt.reshape((-1, 1))),
            #                'p': ca.vertcat(u, tvp, p, Q.reshape((-1, 1)))}

            # look: line 275 in simulator.py

            dae ={
                'x': ca.vertcat(x, P.reshape((-1, 1))),
                'z': z,
                'p': ca.vertcat(u, tvp, p, Q.reshape((-1, 1))),
                #'p': ca.vertcat(u, z, tvp, p, w, Q.reshape((-1, 1))),
                'ode': ca.vertcat(f, dP_dt.reshape((-1, 1))),
                'alg': alg,
            }

            #x0, u_next, z0, tvp0, p0, w0, v0
            #x0, u_next, z0, tvp0, p0, w0

            self.x_p_integrator = ca.integrator('x_integrator', 'idas', dae, opts)

        # setting up integrator for model.model_type = 'discrete'
        elif self.model.model_type == 'discrete':
            if self.model.n_z > 0:
                # Build the DAE function
                #nlp = {'x': sim_z['_z'], 'p': castools.vertcat(sim_x['_x'], sim_p), 'f': castools.DM(0), 'g': alg}
                nlp = {'x': z, 'p': castools.vertcat(x, p, tvp, u, w), 'f': castools.DM(0), 'g': alg}

                self.discrete_dae_solver = castools.nlpsol('dae_roots', 'ipopt', nlp)

        # only for continious case:
        self._check_validity()

        # setting up counter
        self.counter = 0

        # generating flags
        self.flags.update({
            'setup': True,
        })

        # end of function
        return None

    def set_initial_guess(self):

        # checks to ensure proper usage
        assert self.flags['setup'] == True, 'EKF was not setup yet. Please call EKF.setup().'

        # set initial value
        self.x0 = ca.DM(self.x0).full()

        # Covariance Matrix of initial error
        #self.P0 = self.Q

        # setting initial guess for simulator
        #self.simulator.x0 = self.x0
        #self.simulator.set_initial_guess()

        # changing flag
        self.flags['set_initial_guess'] = True

        # return
        return None

    def make_step(self, y_next, u_next, Q_k, R_k, debug_flag = True, force_discrete = False):

        # checks to ensure proper usage
        assert self.flags['setup'] == True, 'EKF was not setup yet. Please call EKF.setup().'
        assert self.flags[
                   'set_initial_guess'] == True, 'Initial guess was not provided. Please call EKF.set_initial_guess().'



        #x, u, z, tvp, p, w = self.model['x', 'u', 'z', 'tvp', 'p', 'w']
        # storing values temporarily
        t0 = self._t0
        tvp0 = self.tvp_fun(t0)
        p0 = self.p_fun(t0)
        x0 = ca.DM(self.x0).full()
        z0 = self.z0
        v0 = np.zeros((self.model.n_y, 1))
        #w0 = np.zeros((0, 1))
        w0 = self.model['w'](0)

        self.counter += 1
        self._t0 += self.settings.t_step

        # Linearisation
        #A, B, C, D = self.model.get_linear_system_matrices(xss=x0, uss=u_next, p=p0, tvp=tvp0)

        #A_k = self.model.A_fun(x0, u_next, z0, tvp0, p0, w0)
        #B_k = self.model.B_fun(x0, u_next, z0, tvp0, p0, w0)
        #C_k = self.model.C_fun(x0, u_next, z0, tvp0, p0, v0)
        #D_k = self.model.D_fun(x0, u_next, z0, tvp0, p0, v0)

        A_k = self.A_fun(x0, u_next, p0, tvp0)
        B_k = self.B_fun(x0, u_next, p0, tvp0)
        C_k = self.C_fun(x0, u_next, p0, tvp0)
        D_k = self.D_fun(x0, u_next, p0, tvp0)

        if self.flags['first_step']:
            #self.P0 = Q_k
            self.P0 = np.eye((self.model.n_x))
            z0 = np.zeros((self.model.n_z))

            self.flags.update({
            'first_step': False,
            })

        # Debug printouts
        print() if debug_flag else None
        print() if debug_flag else None
        print('#######Iteration number####### : ', self.counter) if debug_flag else None

        print('Predicted state (aposteriori) : ', x0) if debug_flag else None
        print('Predicted state (aposteriori) type: ', type(x0)) if debug_flag else None
        print('Predicted state (aposteriori) shape : ', x0.shape) if debug_flag else None

        print('Input : ', u_next) if debug_flag else None
        print('Input type: ', type(u_next)) if debug_flag else None
        print('Input shape : ', u_next.shape) if debug_flag else None

        print() if debug_flag else None
        print('system matrix: A') if debug_flag else None
        print(A_k) if debug_flag else None
        print('Shape:', A_k.shape) if debug_flag else None

        print() if debug_flag else None
        print('input matrix: B') if debug_flag else None
        print(B_k) if debug_flag else None
        print('Shape:', B_k.shape) if debug_flag else None

        print() if debug_flag else None
        print('measurement matrix: C') if debug_flag else None
        print(C_k) if debug_flag else None
        print('Shape:', C_k.shape) if debug_flag else None

        print() if debug_flag else None
        print('output disturbance matrix: D') if debug_flag else None
        print(D_k) if debug_flag else None
        print('Shape:', D_k.shape) if debug_flag else None



        # Apriori and Prediction covariance
        if self.model.model_type is 'continuous' and force_discrete is False:
            initial_cond_x = ca.vertcat(self.x0, self.P0.reshape((-1,1)))
            initial_cond_z = z0
            initial_cond_0 = {
                'x0': initial_cond_x,
                'z0': initial_cond_z
            }
            sol = self.x_p_integrator(x0=initial_cond_x, p=ca.vertcat(u_next, tvp0, p0, Q_k.reshape((-1, 1))))
            #sol = self.x_p_integrator(x0=initial_cond_0, p=ca.vertcat(u_next, z0, tvp0, p0, w0, Q_k.reshape((-1, 1))))
            #u, z, tvp, p, w, Q.reshape((-1, 1))
            x_apriori = sol['xf'].full()[0:self.model.n_x]
            self.P0 = sol['xf'].full()[self.model.n_x:].reshape((self.model.n_x, self.model.n_x))
            y_apriori = self.h_fun(x_apriori, u_next, p0, tvp0, v0)

            print('Continious error preddiction covariance : ', self.P0) if debug_flag else None

        else:
            if self.model.n_z > 0: # Solve DAE only when it exists ...
                #x, p, tvp, u, w
                r = self.discrete_dae_solver(x0 = z0, ubg = 0, lbg = 0, p=castools.vertcat(x0, p0, tvp0, u_next, w0))
                #sim_z_num.master = r['x']
                z0 = r['x']
            x_apriori = self.model._rhs_fun(x0, u_next, z0, tvp0, p0, w0)
            y_apriori = self.model._meas_fun(x_apriori, u_next, z0, tvp0, p0, v0)

            self.P0 = A_k @ self.P0 @ A_k.T + Q_k
            print('Discrete error preddiction covariance : ', self.P0) if debug_flag else None

        # Kalman gain
        L = self.P0 @ C_k.T @ ca.inv_minor(C_k @ self.P0 @ C_k.T + R_k)
        print('Kalman gain : ', L) if debug_flag else None

        # Aposteriori
        x0 = x_apriori + L @ (y_next - y_apriori)
        print('Aposteriori : ', x0) if debug_flag else None

        # Updated error covariance
        self.P0 = (np.eye(self.model.n_x) - L @ C_k) @ self.P0
        print('Updated error preddiction covariance : ', self.P0) if debug_flag else None

        # store current state
        self.x0 = ca.DM(x0).full()
        self.z0 = z0

        # Update data object:
        self.data.update(_x = ca.DM(x0).full())
        self.data.update(_u = u_next)
        #self.data.update(_z = self.simulator.sim_z_num['_z'].full())
        self.data.update(_p = p0)
        self.data.update(_tvp = tvp0)
        self.data.update(_time = t0)
        #self.data.update(_aux = self.simulator.data._z[-1])

        # return
        return ca.DM(x0).full()
    
    def get_p_template(self)->Union[castools.structure3.SXStruct,castools.structure3.MXStruct]:
        """Obtain output template for :py:func:`set_p_fun`.
        Use this method in conjunction with :py:func:`set_p_fun`
        to define the function for retrieving the parameters at each sampling time.

        See :py:func:`set_p_fun` for more details.

        Returns:
            numerical CasADi structure
        """
        return self.model._p(0)


    def set_p_fun(self,p_fun:Callable[[float],Union[castools.structure3.SXStruct,castools.structure3.MXStruct]])->None:
        """Method to set the function which gives the values of the parameters.
        This function must return a CasADi structure which can be obtained with :py:func:`get_p_template`.

        **Example**:

        In the :py:class:`do_mpc.model.Model` we have defined the following parameters:

        ::

            Theta_1 = model.set_variable('parameter', 'Theta_1')
            Theta_2 = model.set_variable('parameter', 'Theta_2')
            Theta_3 = model.set_variable('parameter', 'Theta_3')

        To integrate the ODE or evaluate the discrete dynamics, the simulator needs
        to obtain the numerical values of these parameters at each timestep.
        In the most general case, these values can change,
        which is why a function must be supplied that can be evaluted at each timestep to obtain the current values.

        **do-mpc** requires this function to have a specific return structure which we obtain first by calling:

        ::

            p_template = simulator.get_p_template()

        The parameter function can look something like this:

        ::

            p_template['Theta_1'] = 2.25e-4
            p_template['Theta_2'] = 2.25e-4
            p_template['Theta_3'] = 2.25e-4

            def p_fun(t_now):
                return p_template

            simulator.set_p_fun(p_fun)

        which results in constant parameters.

        A more "interesting" variant could be this random-walk:

        ::

            p_template['Theta_1'] = 2.25e-4
            p_template['Theta_2'] = 2.25e-4
            p_template['Theta_3'] = 2.25e-4

            def p_fun(t_now):
                p_template['Theta_1'] += 1e-6*np.random.randn()
                p_template['Theta_2'] += 1e-6*np.random.randn()
                p_template['Theta_3'] += 1e-6*np.random.randn()
                return p_template

        Args:
            p_fun: A function which gives the values of the parameters

        Raises:
            assert: p must have the right structure
        """
        assert isinstance(p_fun(0), castools.structure3.DMStruct), 'p_fun has incorrect return type.'
        assert self.get_p_template().labels() == p_fun(0).labels(), 'Incorrect output of p_fun. Use get_p_template to obtain the required structure.'
        self.p_fun = p_fun
        self.flags['set_p_fun'] = True

    def get_tvp_template(self)->Union[castools.structure3.SXStruct,castools.structure3.MXStruct]:
        """Obtain the output template for :py:func:`set_tvp_fun`.
        Use this method in conjunction with :py:func:`set_tvp_fun`
        to define the function for retrieving the time-varying parameters at each sampling time.

        Returns:
            numerical CasADi structure
        """
        return self.model._tvp(0)


    def set_tvp_fun(self,tvp_fun:Callable[[float],Union[castools.structure3.SXStruct,castools.structure3.MXStruct]])->None:
        """Method to set the function which returns the values of the time-varying parameters.
        This function must return a CasADi structure which can be obtained with :py:func:`get_tvp_template`.

        In the :py:class:`do_mpc.model.Model` we have defined the following parameters:

        ::

            a = model.set_variable('_tvp', 'a')

        The integrate the ODE or evaluate the discrete dynamics, the simulator needs
        to obtain the numerical values of these parameters at each timestep.
        In the most general case, these values can change,
        which is why a function must be supplied that can be evaluted at each timestep to obtain the current values.

        **do-mpc** requires this function to have a specific return structure which we obtain first by calling:

        ::

            tvp_template = simulator.get_tvp_template()

        The time-varying parameter function can look something like this:

        ::

            def tvp_fun(t_now):
                tvp_template['a'] = 3
                return tvp_template

            simulator.set_tvp_fun(tvp_fun)

        which results in constant parameters.

        Note:
            From the perspective of the simulator there is no difference between
            time-varying parameters and regular parameters. The difference is important only
            for the MPC controller and MHE estimator. These methods consider a finite sequence
            of future / past information, e.g. the weather, which can change over time.
            Parameters, on the other hand, are constant over the entire horizon.

        Args:
            tvp_fun: Function which gives the values of the time-varying parameters

        Raises:
            assertion: tvp_fun has incorrect return type.
            assertion: Incorrect output of tvp_fun. Use get_tvp_template to obtain the required structure.
        """
        assert isinstance(tvp_fun(0), castools.structure3.DMStruct), 'tvp_fun has incorrect return type.'
        assert self.get_tvp_template().labels() == tvp_fun(0).labels(), 'Incorrect output of tvp_fun. Use get_tvp_template to obtain the required structure.'
        self.tvp_fun = tvp_fun

        self.flags['set_tvp_fun'] = True
