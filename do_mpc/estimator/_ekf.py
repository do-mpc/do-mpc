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
from ._estimatorsettings import EstimatorSettings


class EKF(Estimator):
    """Extended Kalman Filter. Setup this class and use :py:func:`EKF.make_step`
    during runtime to obtain the currently estimated states given the measurements ``y0``.

    Warnings:
        Work in progress.
        The current implementation is not working with DAE systems.
    """
    def __init__(self, model:Union[do_mpc.model.Model,do_mpc.model.LinearModel]):
        
        # init
        Estimator.__init__(self, model)
        self.settings = EstimatorSettings()

        # generating flags
        self.flags = {
            'setup': False,
            'set_initial_guess': False,
            'set_tvp_fun': False,
            'set_p_fun': False,
            'first_step': True
        }

        # Initialize structure for initial conditions:
        self._P0 = np.eye(self.model.n_x)


    @property
    def P0(self):
        """Initial error covariance matrix for the Extended Kalman Filter.
        
        This matrix represents the initial uncertainty in the state estimates.
        It must be a positive semi-definite matrix of shape (n_x, n_x).
        
        The default initialization is an identity matrix.
        """
        return self._P0

    @P0.setter
    def P0(self, val):
        """Set the initial error covariance matrix.
        
        Args:
            val: New covariance matrix. Must be a numpy.ndarray with shape (n_x, n_x).
        
        Raises:
            TypeError: If val is not a numpy.ndarray
            ValueError: If matrix has wrong dimensions or is not square
            Warning: If matrix is not symmetric (will be made symmetric)
        """
        # Only accept numpy arrays
        if not isinstance(val, np.ndarray):
            raise TypeError(f"P0 must be a numpy.ndarray, got {type(val).__name__}")
        
        # Validate dimensions
        if val.ndim != 2:
            raise ValueError(f"P0 must be a 2D matrix, got {val.ndim}D array")
        
        if val.shape[0] != val.shape[1]:
            raise ValueError(f"P0 must be square, got shape {val.shape}")
        
        if val.shape[0] != self.model.n_x:
            raise ValueError(
                f"P0 must have shape ({self.model.n_x}, {self.model.n_x}) "
                f"to match state dimension, got {val.shape}"
                )
        self._P0 = ca.DM(val).full()

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
        """Sets up the EKF and finalizes the EKF configuration.
        Only after the setup, the :py:func:`make_step` method becomes available.

        Raises:
            assertion: number of algebraic equations must be zero
        """

        assert self.model._alg.shape[0] == 0, 'EKF with algebraic equations not ready for use!'

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

            # init covariance matrix (symbolic)
            P = ca.SX.sym('P', self.model.n_x, self.model.n_x)
            Q = ca.SX.sym('Q', self.model.n_x, self.model.n_x)

            # covariance integration equation
            dP_dt = (A @ P) + (P @ A.T) + Q

            # setting up dae
            dae ={
                'x': ca.vertcat(x, P.reshape((-1, 1))),
                'z': z,
                'p': ca.vertcat(u, tvp, p, Q.reshape((-1, 1))),
                #'p': ca.vertcat(u, z, tvp, p, w, Q.reshape((-1, 1))),
                'ode': ca.vertcat(f, dP_dt.reshape((-1, 1))),
                'alg': alg,
            }

            # setting up the state covariance integrator, giving t0=0.0 due to deprecated opts dict of casadi
            t0 = 0.0
            self.x_p_integrator = ca.integrator('x_integrator', 'idas', dae, t0, self.settings.t_step)

        # only for continious case:
        self._check_validity()

        # setting up counter
        self.counter = 0

        if not hasattr(self, "P0"):
            # P0 doesn’t exist yet, so initialize it now
            self.P0 = np.eye(self.model.n_x)

        # initialising a default algebraic initial state
        self.z0 = np.zeros((self.model.n_z))

        # updating flags
        self.flags.update({
            'setup': True,
        })

        # end of function
        return None

    def set_initial_guess(self):
        """Initial guess for DAE variables.
        Use the current class attribute :py:attr:`z0` to create the initial guess for the DAE algebraic equations.

        The simulator uses "warmstarting" to solve the continous/discrete DAE system by using the previously computed
        algebraic states as an initial guess. Thus, this method is typically only invoked once.

        Warnings:
            If no initial values for :py:attr:`z0` were supplied during setup, they default to zero.
            If no initial values for :py:attr:`P0` were supplied during setup, they default to a unit matrix.
        """

        # checks to ensure proper usage
        assert self.flags['setup'] == True, 'EKF was not setup yet. Please call EKF.setup().'


        # set initial value for state
        self.x0 = ca.DM(self.x0).full()


        # updating flag
        self.flags['set_initial_guess'] = True

        # return
        return None

    def make_step(self, y_next, u_next, Q_k, R_k):
        """Main method of the EKF class during control runtime. This method is called at each timestep
        and computes the next state estimates :py:obj:`x0`. The method returns the resulting states.

        The initial state :py:attr:`x0` is stored as a class attribute. Use this attribute :py:attr:`x0` to change the initial state, by calling
        :py:func:`set_initial_guess`.

        Args:
            y_next: Current measurement of system.
            u_next: Current input to the system.
            Q_k: Current process noise covariance.
            R_k: Current mesusrement noise covariance.

        Returns:
            x0

        Raises:
            AssertionError: If the EKF was not setup yet or if the initial guess was not set.
            AssertionError: If the dimensions of Q_k and R_k are not correct.
        """

        # checks to ensure proper usage
        assert self.flags['setup'] == True, 'EKF was not setup yet. Please call EKF.setup().'
        assert self.flags[
                   'set_initial_guess'] == True, 'Initial guess was not provided. Please call EKF.set_initial_guess().'
        
        # checks correct dimensions of Q_k and R_k
        assert Q_k.shape == (self.model.n_x, self.model.n_x), 'Q_k must be a square matrix of shape ({}, {})'.format(self.model.n_x, self.model.n_x)
        assert R_k.shape == (self.model.n_y, self.model.n_y), 'R_k must be a square matrix of shape ({}, {})'.format(self.model.n_y, self.model.n_y)

        if self.flags['first_step']:
            self.flags.update({
            'first_step': False,
            })

        # init
        t0 = self._t0
        tvp0 = self.tvp_fun(t0)
        p0 = self.p_fun(t0)
        x0 = ca.DM(self.x0).full()
        z0 = self.z0
        v0 = np.zeros((self.model.n_y, 1))
        w0 = self.model['w'](0)
        P0 = self.P0

        # counter and timer calculation
        self.counter += 1
        self._t0 += self.settings.t_step

        # Linearised system matrices
        A_k = self.A_fun(x0, u_next, p0, tvp0)
        B_k = self.B_fun(x0, u_next, p0, tvp0)
        C_k = self.C_fun(x0, u_next, p0, tvp0)
        D_k = self.D_fun(x0, u_next, p0, tvp0)

        # Apriori and Prediction covariance
        if self.model.model_type is 'continuous':
            initial_cond_x = ca.vertcat(self.x0, P0.reshape((-1,1)))
            initial_guess_z = z0
            sol = self.x_p_integrator(x0 = initial_cond_x, z0 = initial_guess_z, p=ca.vertcat(u_next, tvp0, p0, Q_k.reshape((-1, 1))))
            x_apriori = sol['xf'].full()[0:self.model.n_x]
            P0 = sol['xf'].full()[self.model.n_x:].reshape((self.model.n_x, self.model.n_x))
            y_apriori = self.h_fun(x_apriori, u_next, p0, tvp0, v0)

        else:            
            x_apriori = self.model._rhs_fun(x0, u_next, z0, tvp0, p0, w0)
            y_apriori = self.model._meas_fun(x_apriori, u_next, z0, tvp0, p0, v0)

            P0 = A_k @ P0 @ A_k.T + Q_k

        # Kalman gain
        L = P0 @ C_k.T @ ca.inv_minor(C_k @ P0 @ C_k.T + R_k)

        # Aposteriori
        x0 = x_apriori + L @ (y_next - y_apriori)

        # Updated error covariance
        P0 = (np.eye(self.model.n_x) - L @ C_k) @ P0

        # store current state
        self.x0 = ca.DM(x0).full()
        self.z0 = z0

        # store current covariance matrix
        self.P0 = P0.full()

        # Update data object:
        self.data.update(_x = ca.DM(x0).full())
        self.data.update(_u = u_next)
        self.data.update(_p = p0)
        self.data.update(_tvp = tvp0)
        self.data.update(_time = t0)

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
