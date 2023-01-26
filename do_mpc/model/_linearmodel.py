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
from casadi import *
from casadi.tools import *
import pdb
import warnings
from scipy.signal import cont2discrete
from . import Model

# Define what is included in the Sphinx documentation.
__all__ = ['LinearModel']


class LinearModel(Model):
    """The **do-mpc** LinearModel class. This class is inherited from **do-mpc** model class. 
    This class holds the full model description and is at the core of
    :py:class:`do_mpc.simulator.Simulator`, :py:class:`do_mpc.controller.MPC`, :py:class:`do_mpc.controller.LQR` and :py:class:`do_mpc.estimator.Estimator`.
    This class can be used to define the linear time invariant models in both
    continuous and discrete time.
    The :py:class:`LinearModel` class is created with setting the ``model_type`` (continuous or discrete).
    
    A ``continous`` linear model consists of an underlying ordinary differential equation (ODE)
    
    .. math::
        
        \\dot{x}(t) &= Ax(t)+Bu(t),\\\\
            y &= Cx(t)+Du(t)
            
    whereas a ``discrete`` linear model consists of a difference equation.
        
    .. math::
             
        x_{k+1} &= Ax_k+Bu_k,\\\\
       y_k &= Cx_k+Du_k
       
    The **do-mpc** linear model can be initiated with ``SX`` variable type.
    
    .. note::

        The option ``symvar_type`` will be inherited to all derived classes (e.g. :py:class:`do_mpc.simulator.Simulator`,
        :py:class:`do_mpc.controller.MPC` and :py:class:`do_mpc.estimator.Estimator`).
        All symbolic variables in these classes will be chosen respectively.
        
    **Configuration and setup:**
    Configuring and setting up the :py:class:`LinearModel` involves the following steps:
    
    Model can be setup in two different ways. The first method is as follows:
    
        1. Use :py:func:`set_variable` to introduce new variables to the linear model.
    
        2. Optionally introduce "auxiliary" expressions as functions of the previously defined variables with :py:func:`set_expression`. The expressions can be used for monitoring or be reused as constraints, the cost function etc.
    
        3. Optionally introduce measurement equations with :py:func:`set_meas`. The syntax is identical to :py:func:`set_expression`. By default state-feedback is assumed.
    
        4. Define the right-hand-side of the `discrete` or `continuous` model as a function of the previously defined variables with :py:func:`set_rhs`. This method must be called once for each introduced state.
    
        5. Call :py:func:`setup` to finalize the :py:class:`LinearModel`. No further changes are possible afterwards.
        
    The second method is as follows:
        
        1. Use :py:func:`set_variable` to introduce new variables to the linear model.
        
        2. Optionally introduce "auxiliary" expressions as functions of the previously defined variables with :py:func:`set_expression`. The expressions can be used for monitoring or be reused as constraints, the cost function etc.
        
        3. Call :py:func:`setup` and pass the system dynamics matrices as arguments instead of setting up the right hand side equations and measurement equations to finalize the :py:class:`LinearModel`. No further changes are possible afterwards.
        
    .. note::

        All introduced model variables are accessible as **Attributes** of the :py:class:`Model`.
        Use these attributes to query to variables, e.g. to form the cost function in a seperate file for the MPC configuration.
        
    :param model_type: Set if the model is ``discrete`` or ``continuous``.
    :type model_type: str
    :param symvar_type: Set if the model is configured with CasADi ``SX`` variables (default).
    :type symvar_type: str

    :raises assertion: model_type must be string
    :raises assertion: model_type must be either discrete or continuous

    """
    def __init__(self, model_type=None, symvar_type='SX'):
        super().__init__(model_type, symvar_type)
        if symvar_type == 'MX':
            raise ValueError("class LinearModel can be initialized only with SX variable.")


    @property
    def sys_A(self):
        """State matrix.
        This property provides the state matrix in the numerical array format. Accessible only after model is setup.
        
        :return: A - State matrix
        :rtype: numpy.ndarray
        """
        assert self.flags['setup'] == True, 'Attributes are available after the model is setup.'
        return self._A
    
    @property
    def sys_B(self):
        """Input matrix.
        This property provides the input matrix in the numerical array format. Accessible only after model is setup.
        
        :return: B - Input matrix
        :rtype: numpy.ndarray
        """
        assert self.flags['setup'] == True, 'Attributes are available after the model is setup.'
        return self._B
    
    @property
    def sys_C(self):
        """Output matrix.
        This property provides the output matrix in the numerical array format. Accessible only after model is setup.
        
        :return: C - Output matrix
        :rtype: numpy.ndarray
        """
        assert self.flags['setup'] == True, 'Attributes are available after the model is setup.'
        return self._C
    
    @property
    def sys_D(self):
        """Feedforward matrix.
        This property provides the feedforward matrix in the numerical array format. Accessible only after model is setup.
        
        :return: D - Feedforward matrix
        :rtype: numpy.ndarray
        """
        assert self.flags['setup'] == True, 'Attributes are available after the model is setup.'
        return self._D


    def set_rhs(self, name, rhs):
        """
        Checks if the right-hand-side function is linear and calls :meth:`Model.set_rhs`.
        
        :param name: Reference to previously introduced state names (with :py:func:`LinearModel.set_variable`)
        :type name: sting
        :param rhs: CasADi SX function depending on ``_x``, ``_u``, ``_tvp``, ``_p``.
        :type rhs: casADi SX
        """
        # Check if expression is linear
        if evalf(jacobian(rhs, vertcat(self.x, self.u))).is_constant():
            super(LinearModel, self).set_rhs(name, rhs, process_noise=True)
        else:
            raise ValueError("Given rhs is not linear.")

    def set_meas(self, name, meas):
        """
        Checks if the measurement function is linear and calls :meth:`Model.set_meas`.
        
        :param name: Arbitrary name for the given expression. Names are used for key word indexing.
        :type name: sting
        :param meas: CasADi SX function depending on ``_x``, ``_u``, ``_tvp``, ``_p``.
        :type meas: casADi SX
        """
        # Check if expression is linear
        if evalf(jacobian(meas, vertcat(self.x, self.u))).is_constant():
            super(LinearModel, self).set_meas(name, meas, meas_noise=True)

    def set_alg(self, expr_name, expr, *args, **kwargs):
        """
        .. warning::

            This method is not supported for linear models.
        """
        raise NotImplementedError('Algebraic variables are not supported for linear models.')
        
    def setup(self,A=None, B=None,C=None,D=None):
        """Setup method must be called to finalize the modelling process.
        All required model variables must be declared.
        The right hand side expression for ``_x`` can be set with :py:func:`set_rhs` or can be set by passing the state matrix and input matrix in :py:func:`setup`.

        Sets default measurement function (state feedback) if :py:func:`set_meas` was not called or output matrix, feedforward matrix are not passed in :py:func:`setup`.

        .. warning::

            After calling :py:func:`setup`, the model is locked and no further variables,
            expressions etc. can be set.

        :raises assertion: Definition of right hand side (rhs) is incomplete
        
        :param name: A - State matrix (optional)
        :type name: numpy.ndarray
        :param name: B - Input matrix (optional)
        :type name: numpy.ndarray
        :param name: C - Output matrix (optional)
        :type name: numpy.ndarray
        :param name: D - Feedforward matrix (optional)
        :type name: numpy.ndarray

        :return: None
        :rtype: None
        """
        if not isinstance(A, (np.ndarray, type(None))):
            raise ValueError('A must be a numpy array or None')
        if not isinstance(B, (np.ndarray, type(None))):
            raise ValueError('B must be a numpy array or None')
        if not isinstance(C, (np.ndarray, type(None))):
            raise ValueError('C must be a numpy array or None')
        if not isinstance(D, (np.ndarray, type(None))):
            raise ValueError('D must be a numpy array or None')
    


        # Three use cases:
        # 1. C / D are given -> Create measurement function
        # 2. set_meas has been called -> Measurement function already exists
        # 3. Neither C / D nor set_meas -> No measurement super().setup() creates default measurement function (state feedback)        
        y_meas = None
        if not isinstance(C, type(None)):
            y_meas = C @ self.x.cat
        if not isinstance(D, type(None)):
            y_meas = y_meas + D @ self.u.cat
        if not isinstance(y_meas, type(None)):
            n_y = y_meas.shape[0]
            self.set_meas('y', y_meas)

        n_x = self.x.size
        n_u = self.u.size

        # Create x_next from A and B matrices (if available)
        x_next = None
        if isinstance(A, np.ndarray):
            if A.shape != (n_x, n_x):
                raise ValueError('A must be a square matrix with size n_x x n_x. You have A.shape={}'.format(A.shape))
            else:
                x_next = A @ self.x.cat 
        if isinstance(B, np.ndarray):
            if B.shape != (n_x, n_u):
                raise ValueError('B must be a matrix with size n_x x n_u. You have B.shape={}'.format(B.shape))
            else:
                x_next = x_next + B @ self.u.cat    

        # Set the rhs of the states (if x_next exists)
        if not isinstance(x_next, type(None)):
            for name in self.x.keys():
                ind = self.x.f[name]
                self.set_rhs(name, x_next[ind])


        super(LinearModel, self).setup()

        # Create A,B,C,D matrices (they are not necessarily given) and write them to the class.
        A,B,C,D = self.get_linear_system_matrices()
        self._A = A
        self._B = B
        self._C = C
        self._D = D
        
    def discretize(self, t_step = 0, conv_method = 'zoh'):
        """Converts continuous time to discrete time system.
        
        This method utilizes the exisiting function in scipy library called ``cont2discrete`` to convert continuous time to discrete time system.This method 
        allows the user to specify the type of discretization. For more details about the function `click here`_ .
         
        .. _`click here`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cont2discrete.html
            
        where :math:`A_{\\text{discrete}}` and :math:`B_{\\text{discrete}}` are the discrete state matrix and input matrix repectively and :math:`t_{\\text{sample}}`
        is the sampling time.
        
        .. warning::
            sampling time is zero when not specified or not required
        :param t_step: Sampling time (default - ``0``)
        :type t_step: int or float
        
        :param conv_method: Method of discretization - Five different methods can be applied. (default -`` zoh``)
        :type conv_method: String
        
        :return: model
        :rtype: LinearModel
        """
        assert self.flags['setup'] == True, 'This method can be accessed only after the model is setup using LinearModel.setup().'
        assert self.model_type == 'continuous', 'Given model is already discrete.'

        warnings.warn('sampling time is {}'.format(t_step))
        
        A, B, C, D, t = cont2discrete((self.sys_A,self.sys_B,self.sys_C,self.sys_D), t_step, conv_method)
        
        discreteModel = LinearModel('discrete')

        # Create new variables for linearized model
        self._transfer_variables(self, discreteModel)

        # Setup linearized model
        discreteModel.setup(A,B)
        
        return discreteModel 
        
        
    def get_steady_state(self,xss = None,uss = None):
        """Calculates steady states for the given input or states.
        
        This method calculates steady states of a discrete system for the given steady state input and vice versa.
        The mathematical formulation can be described as 
            
            .. math::
                x_{ss} = (I-A)^{-1}Bu_{ss}\\\\
                
               or\\\\
                
                u_{ss} = B^{-1}(I-A)x_{ss}
        
        :return: Steady state
        :rtype: numpy.ndarray
        
        :return: Steady state input
        :rtype: numpy.ndarray
        """
        #Check whether the model is linear and setup
        assert self.flags['setup'] == True, 'Model is not setup. Please run model.setup() fun to calculate steady state.'
        assert self.model_type == 'discrete', 'Please convert the system to discrete using model.continuous_2_discrete().'
        I = np.identity(np.shape(self.sys_A)[0])
        
        #Calculation of steady state

        if np.all(xss) == None and np.linalg.matrix_rank(self.sys_A) == self.x.shape[0]:
            assert np.all(uss) != None and isinstance(uss,np.ndarray), 'Provide either steady state states or steady state inputs.'
            self.xss = np.linalg.inv(I-self.sys_A)@self.sys_B@uss
            self.uss = uss
            return self.xss
        elif np.all(xss) == None and np.linalg.matrix_rank(self.sys_A) != self.x.shape[0]:
            raise ValueError("State matrix does not have full rank. Hence, either multiple steady state or no steady state values is possible.")
        elif np.all(uss) == None and np.shape(self.sys_B)[0] != np.shape(self.sys_B)[1]:
            assert np.all(xss) != None and isinstance(xss,np.ndarray), 'Provide either steady state states or steady state inputs.'
            self.uss = np.linalg.pinv(self.sys_B)@(I-self.sys_A)@xss
            self.xss = xss
            return self.uss
        elif np.all(uss) == None and np.shape(self.sys_B)[0] == np.shape(self.sys_B)[1]:
            assert np.all(xss) != None and isinstance(xss,np.ndarray), 'Provide either steady state states or steady state inputs.'
            self.uss = np.linalg.inv(self.sys_B)@(I-self.sys_A)@xss
            self.xss = xss
            return self.uss
        
