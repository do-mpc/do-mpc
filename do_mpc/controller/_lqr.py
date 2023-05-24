
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

import casadi.tools as castools
import numpy as np
import warnings
import pdb
import do_mpc.data
from scipy.linalg import solve_discrete_are
from ..model import LinearModel, IteratedVariables
from ._controllersettings import LQRSettings

class LQR(IteratedVariables):
    """Linear Quadratic Regulator.

    .. versionadded:: >v4.5.1

        New interface to settings. The class has an attribute ``settings`` which is an instance of :py:class:`LQRSettings` (please see this documentation for a list of available settings).
        Settings are now chosen as:

        ::

            lqr.settings.t_step = .5
        
        Previously, settings were passed to :py:meth:`set_param`. This method is still available and wraps the new interface.
        The new method has important advantages:
        
        1. The ``lqr.settings`` attribute can be printed to see the current configuration.
        
        2. Context help is available in most IDEs (e.g. VS Code) to see the available settings, the type and a description.


    Use this class to configure and run the LQR controller
    according to the previously configured :py:class:`do_mpc.model.LinearModel` instance. 
    
    Two types of LQR can be desgined:

    1. **Finite Horizon** LQR by choosing, e.g. ``n_horizon = 20``.

    2. **Infinite Horizon** LQR by choosing ``n_horizon = None``.

    The value for ``n_horizon`` is set using :py:meth:`set_param`. 
    
    **Configuration and setup:**
    
    Configuring and setting up the LQR controller involves the following steps:
    
    1. Configure the LQR controller with :py:class:`LQRSettings` class. The LQR instance has the attribute ``settings`` which is an instance of :py:class:`LQRSettings`. 

    2. Set the objective of the control problem with :py:meth:`set_objective`
    
    3. To finalize the class configuration call :py:meth:`setup`.
    
    The :py:class:`LQR` can be used in **two different modes**:

    1. **Standard** mode: 

    - Set set-point with :py:meth:`set_setpoint` (default is ``0``).
    
    - Set ``Q`` and ``R`` values with :py:meth:`set_objective`.
    
    2. **Input Rate Penalization** mode:

    - Setpoint can also be set using :py:meth:`set_setpoint` (default is ``0``).
    
    - Reformulate objective with :py:meth:`set_rterm` to penalize the input rate by setting the value ``delR``.

    - Set ``Q`` and ``R`` values with :py:meth:`set_objective`.
    
    Note:
        The function :py:meth:`set_rterm` mode is not recommended to use if the model is converted from an DAE to an ODE system.
        Because the converted model is already in the rated input formulation. 
    
    Note:
        During runtime call :py:meth:`make_step` with the current state :math:`x` to obtain the optimal control input :math:`u`.
        During runtime call :py:meth:`set_setpoint` with the set points of input :math:`u_{ss}` and states :math:`x_{ss}` in order to update the respective set points.

    Args:
        model : Linear model
    """
    def __init__(self,model:LinearModel):
        self.model = model
        IteratedVariables.__init__(self)
        
        assert isinstance(model, LinearModel), 'LQR can only be used with linear models. Initialize the model with LinearModel class.'
        assert model.flags['setup'] == True, 'Model for LQR was not setup. After the complete model creation call model.setup().'
        assert model.model_type == 'discrete', 'Initialize LQR with discrete system. Discretize the system using LinearModel.discretize()'
        
        self.model_type = model.model_type

        self.data = do_mpc.data.Data(model)
        
        # initialize settings class
        self.settings = LQRSettings()
        
        #Initialize mode of LQR
        self.mode = 'standard'
        
        self.flags = {'setup':False}

    def reset_history(self)->None:
        """Reset the history of the LQR.
        """
        self._t0 = np.array([0])
        self.data.init_storage()
        
    def discrete_gain(self,A:np.ndarray,B:np.ndarray)->np.ndarray:
        """Computes discrete gain. 

        This method computes either the finite horizon discrete gain or infinite horizon discrete gain.
        The gain is computed by the solution of discrete-time algebraic Ricatti equation.
        
        For finite horizon :py:class:`LQR`, the problem formulation is as follows:
            
        .. math::
            \\pi(N) &= P_f\\\\
            K(k) & = -(B'\\pi(k+1)B)^{-1}B'\\pi(k+1)A\\\\
            \\pi(k) & = Q+A'\\pi(k+1)A-A'\\pi(k+1)B(B'\\pi(k+1)B+R)^{-1}B'\\pi(k+1)A
       
        For infinite horizon :py:class:`LQR`, the problem formulation is as follows:
            
        .. math::
            K & = -(B'PB+P)^{-1}B'PA\\\\
            P & = Q+A'PA-A'PB(R+B'PB)^{-1}B'PA\\\\
        
        For example:
            
        ::
                
            K = lqr.discrete_gain(A,B)
        
        Args:
            A : State matrix - constant matrix with no variables
            B : Input matrix - constant matrix with no variables
        
        Returns:
            Gain matrix :math:`K` 
        """
        #Verifying the availability of cost matrices
        assert self.Q.size != 0 and self.R.size != 0 , 'Enter tuning parameter Q and R for the lqr problem using set_objective() function.'
        
        #calculating finite horizon gain
        if self.settings.n_horizon !=None:
            assert self.P.size != 0, 'Terminal cost is required to calculate gain. Enter the required value using set_objective() function.'
            temp_p = self.P
            for k in range(self.settings.n_horizon):
                 K = -np.linalg.inv(np.transpose(B)@temp_p@B+self.R)@np.transpose(B)@temp_p@A
                 temp_pi = self.Q+np.transpose(A)@temp_p@A-np.transpose(A)@temp_p@B@np.linalg.inv(np.transpose(B)@temp_p@B+self.R)@np.transpose(B)@temp_p@A
                 temp_p = temp_pi
            return K
        
        #Calculating infinite horizon gain
        elif self.settings.n_horizon == None:
            pi_discrete = solve_discrete_are(A,B, self.Q, self.R)
            K = -np.linalg.inv(np.transpose(B)@pi_discrete@B+self.R)@np.transpose(B)@pi_discrete@A
            return K
        
    def set_rterm(self,delR:np.ndarray)->None:
        """Modifies the model such that rated input acts as the input. 
        
        Warning:
            Calling :py:meth:`set_rterm`  modifies the objective function
            as well as the state and input matrix.
        
        Warning:
            It is not advisible to execute :py:class:`LQR` in the ``inputRatePenalization`` mode if the model is converted from DAE to ODE system.
            Because the converted model itself is in ``inputRatePenalization`` mode.

        The input rate penalization formulation is given as:
            
        .. math::
            \\begin{aligned}
            x(k+1) &= \\tilde{A} x(k) + \\tilde{B}\\Delta u(k)\\\\
            \\text{where} \\quad
            \\tilde{A} &= \\begin{bmatrix} 
                            A & B \\\\
                            0 & I \\end{bmatrix},\\quad
            \\tilde{B} = \\begin{bmatrix} B \\\\
                         I \\end{bmatrix}
            \\end{aligned}
                            
        We introduce new states of this system as :math:`\\tilde{x} = [x,u]` 
        where :math:`x` and :math:`u` are the original states and input of the system.
        After reformulating the system with :py:meth:`set_rterm`, the discrete gain is calculated
        using :py:meth:`discrete_gain`.
        
        As the system state matrix and input matrix are altered,
        cost matrices are also modified accordingly:
            
        .. math::
            \\tilde{Q} = \\begin{bmatrix}
                            Q & 0 \\\\
                            0 & R \\end{bmatrix},\\quad
            \\tilde{R} = \\Delta R
        
        :param delR: Rated input cost matrix - constant matrix with no variables
        """
        
        #Modifying A and B matrix for input rate penalization
        identity_u = np.identity(np.shape(self.model._B)[1])
        zeros_A = np.zeros((np.shape(self.model._B)[1],np.shape(self.model._A)[1]))
        self.A_rated = np.block([[self.model._A,self.model._B],[zeros_A,identity_u]])
        self.B_rated = np.block([[self.model._B],[identity_u]])
        
        self.delR = delR
        self.mode = 'inputRatePenalization'
    
    def set_param(self,**kwargs)->None:
        """Set the parameters of the :py:class:`LQR` class. Parameters must be passed as pairs of valid keywords and respective argument.
        
        Two different kinds of LQR can be desgined. In order to design a finite horizon LQR, ``n_horizon`` and to design a infinite horizon LQR, ``n_horizon`` 
        should be set to ``None`` (default value).

        .. deprecated:: >v4.5.1
            This function will be deprecated in the future
        
        Warnings:
            This method will be depreciated in a future version. Please set parameters via :py:class:`do_mpc.controller.LQRSettings`.
        
        Note:
            A comprehensive list of all available parameters can be found in :py:class:`do_mpc.controller.LQRSettings`. 

        For example:
        
        ::

            lqr.settings.n_horizon = 20
        
        The old interface, as shown in the example below, can still be accessed until further notice.
        
        For example:

        ::

            lqr.set_param(n_horizon = 20)


        .. _`more details here`: https://codeyarns.github.io/tech/2012-04-25-unpack-operator-in-python.html

        Note:
            The only required parameters  are ``n_horizon``. All other parameters are optional.
            :py:meth:`set_param` can be called multiple times. Previously passed arguments are overwritten by successive calls.
        """
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                    setattr(self.settings, key, value)
            else:
                print('Warning: Key {} does not exist for MPC.'.format(key))
                
    def make_step(self,x0:np.ndarray)->np.ndarray:
        """Main method of the class during runtime. This method is called at each timestep
        and returns the control input for the current initial state.
         
        Args:
            x0 : Current state of the system.
        
        Returns:
            u0 - current input of the system
        """
        #verify setup of lqr is done
        assert self.flags['setup'] == True, 'LQR is not setup. run setup() function.'
        
        # Check input type.
        if isinstance(x0, (np.ndarray, castools.DM)):
            pass
        elif isinstance(x0, castools.structure3.DMStruct):
            x0 = x0.cat
        else:
            raise Exception('Invalid type {} for x0. Must be {}'.format(type(x0), (np.ndarray, castools.DM, castools.structure3.DMStruct)))

        #setting setpoints
        if not hasattr(self, "xss") and not hasattr(self,"uss"):
            self.set_setpoint()
        
        u_prev = self.u0.cat.full()
        #Calculate u in set point tracking mode
        if self.mode == "standard":
            if self.xss.size != 0 and self.uss.size != 0:
                u0 = self.K@(x0-self.xss)+self.uss
        
        #Calculate u in input rate penalization mode
        elif self.mode == "inputRatePenalization":
            x0_aug = self._retreive_augmented_states(x0, u_prev)
            u0 = self.K@(x0_aug-self.xss)+self.uss
            u0 = u0 + u_prev


        t0 = self._t0
        # Store solution:
        self.data.update(_x = x0)
        self.data.update(_u = u0)
        self.data.update(_time = t0)

        # Update initial
        self._t0 = self._t0 + self.settings.t_step
        self._x0.master = x0
        self._u0.master = castools.DM(u0)
        
        return u0
        
    def _retreive_augmented_states(self,x,u):
        """Private method.
        
        This method is used to augmented states and inputs for input rate penalization.
        """
        x0 = np.block([[x],[u]])
        return x0
        
    def set_objective(self, Q:np.ndarray, R:np.ndarray, P:np.ndarray = None)->None:
        """Sets the cost matrix for the Optimal Control Problem.
        
        This method sets the inputs, states and algebraic states cost matrices for the given problem.
        
        Since the controller can be operated in two modes. The objective function differes from each other and is as follows
        
        
        
        **Finite Horizon**:
            
        For **set-point tracking** mode:
            
        .. math::

            \\begin{aligned}       
            J &= \\frac{1}{2}\\sum_{k=0} ^{N-1} (x_k - x_{ss})^T Q(x_k-x_{ss})+(u_k-u_{ss})^T R(u_k-u_{ss})\\\\
                    &+ (x_N-x_{ss})^T P(x_N-x_{ss})
            \\end{aligned}
                        
        For **Input Rate Penalization** mode:
            
        .. math::
                
            J = \\frac{1}{2}\\sum_{k=0} ^{N-1} (\\tilde{x}_k - \\tilde{x}_{ss})^T \\tilde{Q}(\\tilde{x}_k-\\tilde{x}_{ss})+\\Delta u_k^T \\Delta R \\Delta u_k 
                + (\\tilde{x}_N-\\tilde{x}_{ss})^TP(\\tilde{x}_N-\\tilde{x}_{ss})
                    
        **Infinite Horizon**:
            
        For **set-point tracking** mode:
            
        .. math::
            
            J = \\frac{1}{2}\\sum_{k=0} ^{\\inf} (x_k - x_{ss})^T Q(x_k-x_{ss})+(u_k-u_{ss})^T R(u_k-u_{ss}) \\quad \\quad \\quad \\quad \\quad \\quad \\quad
                
        For **Input Rate Penalization** mode:
            
        .. math::
            
            J = \\frac{1}{2}\\sum_{k=0} ^{\\inf} (\\tilde{x}_k - \\tilde{x}_{ss})^T \\tilde{Q}(\\tilde{x}_k-\\tilde{x}_{ss})+ \\Delta u_k^T \\Delta R \\Delta u_k \\quad \\quad \\quad \\quad \\quad \\quad \\quad \\quad \\quad \\quad \\quad
        
        where :math:`\\tilde{x} = [x,u]^T` .

        Note:
            For the problem to be solved in ``inputRatePenalization`` mode, ``Q``, ``R`` and ``delR`` should be set.
            ``delR`` is set using :py:meth:`set_rterm`. ``P`` term is set according to the need of the problem.
            
        For example:

        ::
            
            # Values used are to show how to use this function.
            # For ODE models
            lqr.set_objective(Q = np.identity(2), R = np.identity(2), P = np.identity(2))        
        
        Warning:
            ``Q``, ``R``, ``P`` is chosen as matrix of zeros since it is not passed explicitly.
            If ``P`` is not given explicitly, then ``Q`` is chosen as ``P`` for calculating finite discrete gain
            
        Raises:
            exception: Q matrix must be of type class numpy.ndarray
            exception: R matrix must be of type class numpy.ndarray
            exception: P matrix must be of type class numpy.ndarray

        Args:
            Q : State cost matrix
            R : Input cost matrix
            P : Terminal cost matrix (optional) 
        """
        
        #Verify the setup is not complete
        assert self.flags['setup'] == False, 'Objective can not be set after LQR is setup'
        
        #Set Q, R, P
        self.Q = Q
        self.R = R   

        if P is None and self.settings.n_horizon != None:
            self.P = Q
            warnings.warn('P is not given explicitly. Q is chosen as P for calculating finite discrete gain')
        else:
            self.P = P
        
        #Verify shape of Q,R,P
        assert self.Q.shape == (self.model.n_x,self.model.n_x), 'Q must have shape = {}. You have {}'.format((self.model.n_x,self.model.n_x),self.Q.shape)
        assert self.R.shape == (self.model.n_u,self.model.n_u), 'R must have shape = {}. You have {}'.format((self.model.n_u,self.model.n_u),self.R.shape)
        if isinstance(self.Q, (castools.DM, castools.SX, castools.MX)):
            raise Exception('Q matrix must be of type class numpy.ndarray')
        if isinstance(self.R, (castools.DM, castools.SX, castools.MX)):
            raise Exception('R matrix must be of type class numpy.ndarray')
        if self.settings.n_horizon != None and isinstance(self.P, (castools.DM, castools.SX, castools.MX)):
            raise Exception('P matrix must be of type class numpy.ndarray')
        if self.settings.n_horizon != None:
            assert self.P.shape == self.Q.shape, 'P must have same shape as Q. You have {}'.format(P.shape)

    def set_setpoint(self,xss :np.ndarray= None, uss:np.ndarray = None)->None:   
        """Sets setpoints for states and inputs.

        This method can be used to set setpoints for either states or inputs or for both (states and inputs) at each time step. 
        It can be called inside simulation loop to change the set point dynamically.
        
        Note:
            If setpoints is not specifically mentioned it will be set to zero (default).
        
        For example:
            
        ::
            
            # For ODE models
            lqr.set_setpoint(xss = np.array([[10],[15]]) ,uss = np.array([[2],[3]]))

        Args:
            xss : set point for states of the system(optional)
            uss : set point for inputs of the system(optional)
        """
        assert self.flags['setup'] == True, 'LQR is not setup. Run setup() function.'

        # Set or reset xss. Don't change xss if it already exists and uss is not passed
        if isinstance(xss, np.ndarray):
            self.xss = xss
        elif hasattr(self,'xss'):
            pass
        else:
            self.xss = np.zeros((self.model.n_x,1))

        # Set or reset uss. Don't change uss if it already exists and uss is not passed
        if isinstance(uss, np.ndarray): 
            self.uss = uss
        elif hasattr(self,'uss'):
            pass
        else:
            self.uss = np.zeros((self.model.n_u,1))
        
        if self.mode == 'inputRatePenalization':
            self.xss = np.block([[self.xss],[self.uss]])
            self.uss = np.zeros((self.model.n_u,1))
            assert self.xss.shape == (self.model.n_x+self.model.n_u,1), 'xss must be of shape {}. You have {}'.format((self.model.n_x+self.model.n_u,1),self.xss.shape)
        
        if self.mode == 'standard':
            assert self.xss.shape == (self.model.n_x,1), 'xss must be of shape {}. You have {}'.format((self.model.n_x,1),self.xss.shape)
        assert self.uss.shape == (self.model.n_u,1), 'uss must be of shape {}. You have {}'.format((self.model.n_u,1),self.uss.shape)
    
    def setup(self)->None:
        """Prepares :py:class:`LQR` for execution.
        This method initializes and ensures that all the parameters that are necessary to desgin the lqr are available.
        
        Raises:
            exception: mode must be standard, inputRatePenalization, None. you have {string value}

        """
        self.settings.check_for_mandatory_settings()

        if self.mode in ['standard',None]:
            self.K = self.discrete_gain(self.model._A,self.model._B)
        elif self.mode == 'inputRatePenalization':
            #Modifying Q and R matrix for input rate penalization
            zeros_Q = np.zeros((np.shape(self.Q)[0],np.shape(self.R)[1]))
            zeros_Ru = np.zeros((np.shape(self.R)[0],np.shape(self.Q)[1]))
            self.Q = np.block([[self.Q,zeros_Q],[zeros_Ru,self.R]])            
            if self.settings.n_horizon != None:
                self.P = np.block([[self.P,zeros_Q],[zeros_Ru,self.R]])
            self.R = self.delR
            
            if hasattr(self, "A_rated") and hasattr(self, "B_rated"):
                self.K = self.discrete_gain(self.A_rated,self.B_rated)
            else:
                 raise AttributeError("set delR using set_rterm fun to execute in inputRatePenalization mode.")   
        else:
            raise Exception('mode must be standard, inputRatePenalization, None. you have {}'.format(self.method))
        self.flags['setup'] = True