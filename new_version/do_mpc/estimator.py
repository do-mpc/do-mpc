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
import do_mpc.data
import do_mpc.optimizer

class estimator:
    def __init__(self, model):
        self.model = model

        assert model.flags['setup'] == True, 'Model for estimator was not setup. After the complete model creation call model.setup_model().'

        self._x0 = model._x(0.0)
        self._u0 = model._u(0.0)
        self._z0 = model._z(0.0)
        self._t0 = np.array([0.0])


    def set_initial_state(self, x0, reset_history=False):
        """Set the intial state of the estimator.
        Optionally resets the history. The history is empty upon creation of the estimator.

        :param x0: Initial state
        :type x0: numpy array
        :param reset_history: Resets the history of the estimator, defaults to False
        :type reset_history: bool (,optional)

        :return: None
        :rtype: None
        """
        assert x0.size == self.model._x.size, 'Intial state cannot be set because the supplied vector has the wrong size. You have {} and the model is setup for {}'.format(x0.size, self.model._x.size)
        assert isinstance(reset_history, bool), 'reset_history parameter must be of type bool. You have {}'.format(type(reset_history))
        if isinstance(x0, (np.ndarray, casadi.DM)):
            self._x0 = self.model._x(x0)
        elif isinstance(x0, structure3.DMStruct):
            self._x0 = x0
        else:
            raise Exception('x0 must be of tpye (np.ndarray, casadi.DM, structure3.DMStruct). You have: {}'.format(type(x0)))

        if reset_history:
            self.reset_history()

    def reset_history(self):
        """Reset the history of the estimator
        """
        self.data.init_storage()


class state_feedback(estimator):
    def __init__(self, model):
        super().__init__(model)
        self.data = do_mpc.data.observer_data(self.model)

    def make_step(self, y0):
        return y0

class ekf(estimator):
    def __init__(self, model):
        super().__init__(model)
        self.data = do_mpc.data.observer_data(self.model)

class mhe(do_mpc.optimizer):
    def __init__(self, model):
        super().__init__(model)

        # Initialize structures for bounds, scaling, initial values by calling the symbolic structures defined in the model
        # with the default numerical value.
        # This returns an identical numerical structure with all values set to the passed value.
        self._p_scaling = model._p(1.0)

        # Parameters that can be set for the MHE:
        self.data_fields = [
            'n_horizon',
            't_step',
            'state_discretization',
            'collocation_type',
            'collocation_deg',
            'collocation_ni',
            'store_full_solution',
            'store_lagr_multiplier',
            'store_solver_stats',
            'nlpsol_opts'
        ]

        # Default Parameters:
        self.state_discretization = 'collocation'
        self.collocation_type = 'radau'
        self.collocation_deg = 2
        self.collocation_ni = 1
        self.store_full_solution = False
        self.store_lagr_multiplier = True
        self.store_solver_stats = [
            'success',
            't_wall_S',
            't_wall_S',
        ]
        self.nlpsol_opts = {} # Will update default options with this dict.

        # Flags are checked when calling .setup.
        self.flags = {
            'setup': False,
            'set_tvp_fun': False,
            'set_objective': False,
        }

    def set_objective(self, objective):
        assert objective.shape == (1,1), 'objective must have shape=(1,1). You have {}'.format(objective.shape)
        assert self.flags['setup'] == False, 'Cannot call .set_objective after .setup.'

        self.flags['set_objective'] = True
        # TODO: Add docstring
        _x, _u, _z, _tvp, _p, _aux, _y_meas, _y_calc = self.model.get_variables()

        self.mhe_obj_fun = Function('mhe_obj_fun', [_x, _u, _z, _tvp, _p, _y_meas], [objective])

    def get_arrival_weight_template(self):
        return self.model._x(0)


    def set_arrival_weight_fun(self, fun):
        self.arrival_weight_fun = fun


    def setup(self):
        # Create struct for _nl_cons:
        # Use the previously defined SX.sym variables to declare shape and symbolic variable.
        self._nl_cons = struct_SX([
            entry(expr_i['expr_name'], expr=expr_i['expr']) for expr_i in self.nl_cons_list
        ])
        # Make function from these expressions:
        _x, _u, _z, _tvp, _p, _aux, _y = self.model.get_variables()
        self._nl_cons_fun = Function('nl_cons_fun', [_x, _u, _z, _tvp, _p], [self._nl_cons])
        # Create bounds:
        self._nl_cons_ub = self._nl_cons(0)
        self._nl_cons_lb = self._nl_cons(-np.inf)
        # Set bounds:
        for nl_cons_i in self.nl_cons_list:
            self._nl_cons_lb[nl_cons_i['expr_name']] = nl_cons_i['lb']

        self.setup_mhe()
