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



class EKF(Estimator):
    """Extended Kalman Filter. Setup this class and use :py:func:`EKF.make_step`
    during runtime to obtain the currently estimated states given the measurements ``y0``.

    Warnings:
        Work in progress.
    """
    def __init__(self, model:Union[do_mpc.model.Model,do_mpc.model.LinearModel], Q, R):
        
        # init
        Estimator.__init__(self, model)
        self.settings = EstimatorSettings()
        
        # Stores the variances of state(s) and measurement(s)
        self.Q = Q
        self.R = R

        # generating flags
        self.flags = {
            'setup': False,
            'set_initial_guess': False,
            'simulator_setup': False,
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

        self._check_validity()

        # generating flags
        self.flags.update({
            'setup': True,
        })

        self.simulator_setup()

        return None
    
    def simulator_setup(self):

        self.simulator = Simulator(model=self.model)

        self.simulator.set_param(t_step = self.settings.t_step)

        self.simulator.set_p_fun(self.p_fun)
        self.simulator.set_tvp_fun(self.tvp_fun)

        self.simulator.setup()

        self.flags.update({
            'simulator_setup': True,
        })

        return None

    def set_initial_guess(self):

        # checks to ensure proper usage
        assert self.flags['setup'] == True, 'EKF was not setup yet. Please call EKF.setup().'

        # set initial value
        self.x_ekf = ca.DM(self.x0).full()

        # Covariance Matrix of initial error
        self.P0 = self.Q
        self.counter = 0

        # setting initial guess for simulator
        self.simulator.x0 = self.x0
        self.simulator.set_initial_guess()


        # changing flag
        self.flags['set_initial_guess'] = True

        # return
        return None

    def make_step_old(self, y_next, u_next, debug_flag = False, simulate_flag = True):
        
        # exp
        t0 = self._t0
        tvp0 = self.tvp_fun(t0)
        p0 = self.p_fun(t0)

        #self.opt_p_num['_p_est_prev'] = p_est0
        #self.opt_p_num['_p_set'] = p_set0
        #self.opt_p_num['_tvp'] = tvp0['_tvp']

        #self.solve()


        #print("Iternation No:", self.counter)
        self.counter += 1
        print() if debug_flag else None
        print() if debug_flag else None
        print('#######Iteration number####### : ', self.counter) if debug_flag else None

        # checks to ensure proper usage
        assert self.flags['setup'] == True, 'EKF was not setup yet. Please call EKF.setup().'
        assert self.flags['set_initial_guess'] == True, 'Initial guess was not provided. Please call EKF.set_initial_guess().'
        assert self.flags['simulator_setup'] == True, 'EKF Simulator was not setup. Please call EKF.simulator_setup().'


        # Measurement update / Correction
        # needs to be checked
        #A,B,C,D = self.model.get_linear_system_matrices(xss = self.x_ekf, uss = u_next, z = self.model.z, tvp = self.model.tvp, p = self.model.p)
        #A,B,C,D = self.model.get_linear_system_matrices(xss = self.x_ekf, uss = u_next, z = simulator.sim_z_num['_z'].full(), tvp = simulator.sim_p_num['_tvp'].full(), p = simulator.sim_p_num['_p'].full())
        #print(type(self.tvp_fun))
        #print(type(self.p_fun))
        #A,B,C,D = self.model.get_linear_system_matrices(xss = self.x_ekf, uss = u_next, z = simulator.sim_z_num['_z'].full(), tvp = self.tvp_fun, p = self.p_fun)

        print('Apriori state : ', self.x_ekf) if debug_flag else None
        print('Apriori state type: ', type(self.x_ekf)) if debug_flag else None
        print('Apriori state shape : ', self.x_ekf.shape) if debug_flag else None

        A,B,C,D = self.model.get_linear_system_matrices(xss = self.x_ekf, uss = u_next, p = p0, tvp=tvp0)

        #A = ca.DM(A).full()
        #B = ca.DM(B).full()
        #C = ca.DM(C).full()
        #D = ca.DM(D).full()
        print('system matrix: A', A) if debug_flag else None
        print('input matrix: B', B) if debug_flag else None
        print('measurement matrix C: ', B) if debug_flag else None
        print('output disturbance matrix D: ', B) if debug_flag else None

        # Observability 
        assert self.check_obsevability(A,C), 'System not observable. EKF failed!'

        # Optimal Kalman gain
        denominator = C @ self.P0 @ C.T + self.R
        print('denominator: ', denominator) if debug_flag else None
        L = self.P0 @ C.T @ ca.inv_minor(denominator)
        print('Kalman gain: ', L) if debug_flag else None
        
        # Updating observer with Kalman filter (Aposteriori)
        if simulate_flag is False:
            x_ekf_current = self.x_ekf + L @ (y_next - C @ self.x_ekf)
        else:
            y_make_step = self.simulator.make_step(u_next, v0=0*np.random.randn(self.model.n_v,1))
            x_ekf_current = self.x_ekf + L @ (y_next - y_make_step)
            #self.simulator.x0 = x_ekf_current

        # remove
        #x_ekf_current = self.x_ekf + L @ (y_next - C @ self.x_ekf)


        self.x_ekf = x_ekf_current
        print('Aposteriori state (observed state): ', x_ekf_current) if debug_flag else None

        # Updating error covariance matrix aposteriori
        self.P0 = (np.eye(self.model.n_x) - L @ C) @ self.P0
        print('Error covariance matrix : ', self.P0) if debug_flag else None

        # Prediction (Apriori)
        if simulate_flag is False:
            #self.x_ekf = ca.DM((A@self.x_ekf) + (B@u_next)).full()
            self.x_ekf = (A@self.x_ekf) + (B@u_next)
        else:
            # remove
            #y_make_step = self.simulator.make_step(u_next, v0=0*np.random.randn(self.model.n_v,1))
            self.x_ekf = self.simulator.data._x[-1].reshape((-1,1))
        print('Apriori state : ', self.x_ekf) if debug_flag else None

        

        # Updating error covariance matrix of apriori
        self.P0 = ca.DM(A @ self.P0 @ A.T + self.Q).full()
        print('Updated error covariance matrix : ', self.P0) if debug_flag else None


        # Update data object:
        self.data.update(_x = x_ekf_current)
        self.data.update(_u = u_next)
        self.data.update(_z = self.simulator.sim_z_num['_z'].full())
        self.data.update(_p = p_set0)
        self.data.update(_tvp = tvp0['_tvp', -1])
        self.data.update(_time = t0)
        #self.data.update(_aux = self.aux0)

        # return
        return x_ekf_current
    
    def make_step(self, y_next, u_next, debug_flag = False, non_linear_simulation = True):
        
        # storing values temporarily
        t0 = self._t0
        tvp0 = self.tvp_fun(t0)
        p0 = self.p_fun(t0)
        if non_linear_simulation:
            x0 = self.x0
        else:
            x0 = ca.DM(self.x0).full()

        # checks to ensure proper usage
        assert self.flags['setup'] == True, 'EKF was not setup yet. Please call EKF.setup().'
        assert self.flags['set_initial_guess'] == True, 'Initial guess was not provided. Please call EKF.set_initial_guess().'
        assert self.flags['simulator_setup'] == True, 'EKF Simulator was not setup. Please call EKF.simulator_setup().'

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
        
        # Linearisation
        A,B,C,D = self.model.get_linear_system_matrices(xss = x0, uss = u_next, p = p0, tvp = tvp0)

        # Debug printouts
        print() if debug_flag else None
        print('system matrix: A') if debug_flag else None
        print(A) if debug_flag else None
        print('Shape:', A.shape) if debug_flag else None

        print() if debug_flag else None
        print('input matrix: B') if debug_flag else None
        print(B) if debug_flag else None
        print('Shape:', B.shape) if debug_flag else None

        print() if debug_flag else None
        print('measurement matrix: C') if debug_flag else None
        print(C) if debug_flag else None
        print('Shape:', C.shape) if debug_flag else None

        print() if debug_flag else None
        print('output disturbance matrix: D') if debug_flag else None
        print(D) if debug_flag else None
        print('Shape:', D.shape) if debug_flag else None


        # Observability 
        assert self.check_obsevability(A,C), 'System is not observable. EKF failed!'

        # Apriori
        if non_linear_simulation:
            self.simulator.x0 = x0
            y_apriori = self.simulator.make_step(u0 = u_next)
            t0 = self.simulator._t0
            x_apriori = self.simulator.data._x[-1]
        else:
            x_apriori = (A@x0) + (B@u_next)


        # Preddiction covariance
        self.P0 = A @ self.P0 @ A.T + self.Q

        # Kalman gain
        L = self.P0 @ C.T @ ca.inv_minor(C @ self.P0 @ C.T + self.R)

        # Aposteriori
        if non_linear_simulation:
            self.simulator._t0 = t0
            self.simulator.x0 = x_apriori
            y_aposteriori = self.simulator.make_step(u0=u_next)
            x0 = self.x0 + L @ (y_next - y_aposteriori)
        else:
            x0 = self.x0 + L @ (y_next - (C @ x0 + D @ u_next))

        # Updated error covariance
        self.P0 = (np.eye(self.model.n_x) - L @ C) @ self.P0

        # store current state
        self.x0 = ca.DM(x0).full()

        # Update data object:
        self.data.update(_x = self.x0)
        self.data.update(_u = u_next)
        #self.data.update(_z = self.simulator.sim_z_num['_z'].full())
        self.data.update(_p = p0)
        self.data.update(_tvp = tvp0)
        self.data.update(_time = t0)
        #self.data.update(_aux = self.simulator.data._z[-1])

        self.flags.update({
            'first_step': False,
        })

        # return
        return x0

    def check_obsevability(self, A, C):
        """
        obs_mat = C
        print("C:", C)
        for i in range(self.nx-1):
            CAN = C @ np.linalg.matrix_power(A, i+1)
            print(i)
            print(CAN)
            obs_mat = np.vstack([obs_mat, CAN])

        print(obs_mat.shape)
        """
        return True
    
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