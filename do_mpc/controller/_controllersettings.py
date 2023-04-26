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

#import data class decorator
from dataclasses import dataclass, field
from typing import Dict,List

@dataclass
class ControllerSettings:
    """Settings for :py:class:`do_mpc.controller`.

    This class contains the mandaory settings for the all the controllers available int :py:class:`do_mpc.controller`.
    This class creates an instance of type :py:class:`ControllerSettings` and adds it to its class attributes.
    """
    t_step: float = None
    """Timestep of the controller"""

    def check_for_mandatory_settings(self):
        """Method to assert the necessary settings required to design :py:class:`do_mpc.controller`
        """
        if self.t_step is None:
            raise ValueError("t_step must be set")
        
@dataclass
class LQRSettings(ControllerSettings):
    """Settings for :py:class:`do_mpc.controller.LQR`.

    The :py:class:`do_mpc.controller.LQR` automatically creates an instance of type :py:class:`LQRSettings` and adds it to its class attributes.

    Example to change settings:

    ::

        lqr.settings.n_horizon = 20

    Note:     
        Settings cannot be updated after calling :py:meth:`do_mpc.controller.LQR.setup`.
    """
    n_horizon: int = None
    """Prediction horizon of the optimal control problem. 
   Defaults to ``None``, which represents an infinite horizon. 
   """
        
@dataclass
class MPCSettings(ControllerSettings):
    """Settings for :py:class:`do_mpc.controller.MPC`.
    The :py:class:`do_mpc.controller.MPC` automatically creates an instance of type :py:class:`MPCSettings` and adds it to its class attributes.

    Example to change settings:

    ::

        mpc.settings.n_horizon = 20

    Note:     
        Settings cannot be updated after calling :py:meth:`do_mpc.controller.MPC.setup`.
    """
    n_horizon:int = None
    """Prediction horizon of the optimal control problem. 
    
    Parameter must be set by user"""
    n_robust: int = 0
    """Robust horizon for robust scenario-tree MPC.
    
    Note:
        Optimization problem grows exponentially with n_robust.
    """
    open_loop: bool = False
    """Setting for scenario-tree MPC.
    
    Note:
        If the parameter is ``False``, for each timestep AND scenario an individual control input is computed. 
        If set to ``True``, the same control input is used for each scenario.
    """
    use_terminal_bounds: bool = False
    """Choose if terminal bounds for the states are used.

    Set terminal bounds with :py:attr:`do_mpc.controller.MPC.terminal_bounds`.
    """
    state_discretization: str = 'collocation'
    """Choose the state discretization for continuous models. 
    
    Note:
        Currently only ``'collocation'`` is available. Defaults to ``'collocation'``. 
        Has no effect if model is created in discrete type.
    """
    collocation_type: str = 'radau'
    """Choose the collocation type for continuous models with collocation as state discretization. 
    
    Note:
        Currently only ``'radau'`` is available.
    """
    collocation_deg: int = 2
    """Choose the collocation degree for continuous models with collocation as state discretization."""
    collocation_ni: int = 1
    """For orthogonal collocation choose the number of finite elements for the states within a time-step (and during constant control input). 
    
    Can be used to avoid high-order polynomials."""
    nl_cons_check_colloc_points: bool = False
    """For orthogonal collocation choose whether the nonlinear bounds set with :py:func:`do_mpc.controller.MPC.set_nl_cons` are evaluated once per finite Element or for each collocation point."""
    nl_cons_single_slack: bool = False
    """ If ``True``, soft-constraints set with :py:func:`do_mpc.controller.MPC.set_nl_cons` introduce only a single slack variable for the entire horizon."""
    cons_check_colloc_points: bool = True
    """For orthogonal collocation choose whether the linear bounds set with :py:attr:`do_mpc.controller.MPC.bounds` are evaluated once per finite Element or for each collocation point."""
    store_full_solution: bool = False
    """Choose whether to store the full solution of the optimization problem. 
    
    This is required for animating the predictions in post processing. However, it drastically increases the required storage."""
    store_lagr_multiplier: bool = True
    """Choose whether to store the lagrange multipliers of the optimization problem. 
    
    Note:
        Increases the required storage.
    """
    store_solver_stats : List[str] = field(default_factory=lambda:['success','t_wall_total'])
    """Choose which solver statistics to store. 
    
    Must be a list of valid statistics. This attribute is an object of type list."""
    nlpsol_opts: Dict = field(default_factory=dict)
    """Dictionary with options for the CasADi solver call ``nlpsol`` with plugin ``ipopt``. 
    
    All options are listed `here <http://casadi.sourceforge.net/api/internal/d4/d89/group__nlpsol.html>`_."""

    def check_for_mandatory_settings(self):
        """Method to assert the necessary settings required to design :py:class:`do_mpc.controller.MPC`
        """
        super().check_for_mandatory_settings()

        if self.n_horizon is None:
            raise ValueError("n_horizon must be set")
    
    def supress_ipopt_output(self):
        """Method to supress the ipopt solver output.

        This method set the revelvant settings in the ipopt solver in order to supress the output log.
        """
        supress_ipopt = {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0}
        self.nlpsol_opts.update(supress_ipopt)

    def set_linear_solver(self,solver_name: str = "MA27"):
        """Method to set the linear solver to ``MA27``.

        This method enables to set the linear solver to ``MA27``. 
        This change in many cases will drastically boost the speed of do-mpc.

        Example:

        ::

            mpc.settings.set_linear_solver(solver_name = "MA27")
        
        Args:
            solver_name: Specify the linear solver name
        """  
        self.nlpsol_opts['ipopt.linear_solver'] = solver_name

