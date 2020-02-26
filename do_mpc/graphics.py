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
import matplotlib.pyplot as plt
import matplotlib.axes as maxes
from casadi import *
from casadi.tools import *
import pdb


class Graphics:
    """Graphics module to present the results of **do-mpc**.
    The module is independent of all other modules and can be used optionally.
    The module can also be used with pickled result files in post-processing for flexible and custom graphics.

    The graphics module is based on Matplotlib and allows for fully customizable, publication ready graphics and animations.
    User defined graphics are configured prior to plotting results, e.g.:

    ::

        # Initialize graphic:
        graphics = do_mpc.graphics.Graphics()

        # Create figure with arbitrary Matplotlib method
        fig, ax = plt.subplots(5, sharex=True)
        # Configure plot (pass the previously obtained ax objects):
        graphics.add_line(var_type='_x', var_name='C_a', axis=ax[0])
        graphics.add_line(var_type='_x', var_name='C_b', axis=ax[0])
        graphics.add_line(var_type='_x', var_name='T_R', axis=ax[1])
        graphics.add_line(var_type='_x', var_name='T_K', axis=ax[1])
        graphics.add_line(var_type='_aux', var_name='T_dif', axis=ax[2])
        graphics.add_line(var_type='_u', var_name='Q_dot', axis=ax[3])
        graphics.add_line(var_type='_u', var_name='F', axis=ax[4])
        # Optional configuration of the plot(s) with matplotlib:
        ax[0].set_ylabel('c [mol/l]')
        ax[1].set_ylabel('Temperature [K]')
        ax[2].set_ylabel('\Delta T [K]')
        ax[3].set_ylabel('Q_heat [kW]')
        ax[4].set_ylabel('Flow [l/h]')

        fig.align_ylabels()

    After initializing the :py:class:`Graphics` module,
    the :py:func:`Graphics.add_line` method is used to define which results are to be plotted on which existing axes object.
    Note that :py:func:`Graphics.add_line` does not create a graphic or plots results.
    The graphic is obtained with the :py:func:`Graphics.plot_results` method, which takes a :py:class:`do_mpc.data.Data` object as input.
    Each module (e.g.: :py:class:`do_mpc.simulator.Simulator`, :py:class:`do_mpc.estimator.MHE`, :py:class:`do_mpc.controller.MPC`)
    has its own :py:class:`do_mpc.data.Data` object.
    Furthermore, the module contains the :py:func:`Graphics.plot_predictions` method,
    which can be used to show the predicted trajectories for the :py:class:`do_mpc.controller.MPC`.

    .. note::
        A high-level API for obtaining a configured :py:class:`Graphics` module is the :py:func:`default_plot` function.
        Use this function and the obtained :py:class:`Graphics` module in the developement process.

    Animations can be setup with the follwing loop:

    ::

        for k in range(50):
            # do-mpc loop:
            u0 = mpc.make_step(x0)
            y_next = simulator.make_step(u0)
            x0 = estimator.make_step(y_next)

            # Reset and replot results and predictions with Graphics:
            graphics.reset_axes()
            graphics.plot_results(mpc.data, linewidth=3)
            # The second and third argument can be omitted if this information is stored in the data object (optional setting).
            graphics.plot_predictions(mpc.data, mpc.opt_x_num, mpc.opt_aux_num)
            plt.show()
    """
    def __init__(self):
        self.line_list = []
        self.ax_list  = []
        self.color = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def reset_axes(self):
        """Clears the lines on all axes which were passed with :py:func:`Graphics.add_line`.
        Method is called internally, before each plot.
        """
        for ax_i in self.ax_list:
            ax_i.lines = []
        self.reset_prop_cycle()

    def reset_prop_cycle(self):
        """Resets the property cycle for all axes which were passed with :py:func:`Graphics.add_line`.
        The matplotlib color cycler is restarted.
        """
        for ax_i in self.ax_list:
            ax_i.set_prop_cycle(None)

    def add_line(self, var_type, var_name, axis, **pltkwargs):
        """add_line is called during setting up the :py:class:`Graphics` class. This is typically the last step of configuring **do-mpc**.
        Each call of :py:func:`Graphics.add_line` adds a line to the passed axis according to the variable type (``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``, ``_aux``)
        and its name (as defined in the :py:class:`do_mpc.model.Model`).
        Furthermore, all valid matplotlib .plot arguments can be passed as optional keyword arguments, e.g.: ``linewidth``, ``color``, ``alpha``.

        :param var_type: Variable type to be plotted. Valid arguments are ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``, ``_aux``.
        :type var_type: string

        :param var_name: Variable name. Must reference the names defined in the model for the given variable type.
        :type var_name: string

        :param axis: Variable name. Must reference the names defined in the model for the given variable type.
        :type axis: matplotlib.axes.Axes object.

        :param pltkwargs: Valid matplotlib pyplot keyword arguments (e.g.: ``linewidth``, ``color``, ``alpha``)
        :type pltkwargs: optional

        :raises assertion: var_type argument must be a string
        :raises assertion: var_name argument must be a string
        :raises assertion: var_type argument must reference to the valid var_types of do-mpc models.
        :raises assertion: axis argument must be matplotlib axes object.
        """
        assert isinstance(var_type, str), 'var_type argument must be a string. You have: {}'.format(type(var_type))
        assert isinstance(var_name, str), 'var_name argument must be a string. You have: {}'.format(type(var_name))
        assert var_type in ['_x', '_u', '_z', '_tvp', '_p', '_aux'], 'var_type argument must reference to the valid var_types of do-mpc models. Note that _aux_expression are currently not supported for plotting.'
        assert isinstance(axis, maxes.Axes), 'axis argument must be matplotlib axes object.'

        self.line_list.append(
            {'var_type': var_type, 'var_name': var_name, 'ax': axis, 'reskwargs': pltkwargs, 'predkwargs':pltkwargs.copy()}
        )
        self.ax_list.append(axis)


    def plot_results(self, data, t_ind=None, **pltkwargs):
        """Plots the results stored in the passed data object for the plot configuration.
        Each **do-mpc** module has an individual data object.
        Use the ``t_ind`` parameter to plot only until the given time index. This is mostly used in post-processing for animations.
        Optionally pass an arbitrary number of valid pyplot.plot arguments (e.g. ``linewidth``, ``color``, ``alpha``), which is applied to ALL lines.

        :param data: do-mpc data object. Either from unpickled results or the created modules. The data object is updated at each ``make_step``  call.
        :type data: do-mpc data object

        :param t_ind: Plot results up until this time index.
        :type t_ind: int

        :param pltkwargs: Valid matplotlib pyplot keyword arguments (e.g.: ``linewidth``, ``color``, ``alpha``)
        :type pltkwargs: optional

        :raises assertion: t_ind argument must be a int
        :raises assertion: t_ind argument must not exceed the length of the results

        :return: All plotted lines on all supplied axes.
        :rtype:  list
        """
        if t_ind is not None:
            assert isinstance(t_ind, int), 'The t_ind param must be of type int. You have: {}'.format(type(t_ind))
            assert t_ind <= data._time.shape[0], 'The t_ind param must not exceed the length of the results. You choose t_ind={}, where only n={} elements are available.'.format(t_ind, data._time.shape[0])
        # Make index "inclusive", if it is passed. This means that for index 1, the elements at 0 AND 1 are plotted.
        if t_ind is not None:
            t_ind+=1

        self.reset_prop_cycle()
        lines = []
        for line_i in self.line_list:
            line_i['reskwargs'].update(pltkwargs)
            time = data._time[:t_ind]
            res_type = getattr(data, line_i['var_type'])
            # The .f() method returns an index of a casadi Struct, given a name.
            var_ind = data.model[line_i['var_type']].f[line_i['var_name']]
            if line_i['var_type'] in ['_u']:
                lines.extend(line_i['ax'].step(time, res_type[:t_ind, var_ind], **line_i['reskwargs']))
            else:
                lines.extend(line_i['ax'].plot(time, res_type[:t_ind, var_ind], **line_i['reskwargs']))

        return lines

    def plot_predictions(self, data, opt_x_num=None, opt_aux_num=None, t_ind=-1, **pltkwargs):
        """Plots the predicted trajectories for the plot configuration.
        The predicted trajectories are part of the optimal solution at each timestep and can be passed either as the optional
        argument (``opt_x_num``) or they are part of the data structure, if the optimizer was set to store the optimal solution (see :py:func:do_mpc.controller.MPC.set_param)
        The plot predictions method can only be called with data from the :py:class:`do_mpc.controller.MPC` object and raises an error if called with data from other objects.
        Use the ``t_ind`` parameter to plot the prediction for the given time instance. This is mostly used in post-processing for animations.
        Optionally pass an arbitrary number of valid pyplot.plot arguments (e.g. ``linewidth``, ``color``, ``alpha``), which is applied to ALL lines.

        :param data: do-mpc (optimizer) data object. Either from unpickled results or the created modules.
        :type data: do-mpc (optimizer) data object

        :param t_ind: Plot predictions at this time index.
        :type t_ind: int

        :param pltkwargs: Valid matplotlib pyplot keyword arguments (e.g.: ``linewidth``, ``color``, ``alpha``)
        :type pltkwargs: , optional

        :raises assertion: Can only call plot_predictions with data object from do-mpc optimizer
        :raises Exception: Cannot plot predictions if full solution is not stored or supplied when calling the method

        :return: All plotted lines on all supplied axes.
        :rtype:  list
        """
        assert data.dtype == 'MPC', 'Can only call plot_predictions with data object from do-mpc optimizer.'
        assert isinstance(t_ind, int), 'The t_ind param must be of type int. You have: {}'.format(type(t_ind))

        t_now = data._time[t_ind]
        # These fields only exist, if data type (dtype) os optimizer:
        t_step = data.meta_data['t_step']
        n_horizon = data.meta_data['n_horizon']
        structure_scenario = data.meta_data['structure_scenario']

        # Check if full solution is stored in data, or supplied as optional input. Raise error is neither is the case.
        if opt_x_num is None and data.meta_data['store_full_solution']:
            opt_x_num = data.opt_x(data._opt_x_num[t_ind])
        elif opt_x_num is not None:
            pass
        else:
            raise Exception('Cannot plot predictions if full solution is not stored or supplied (opt_x_num) when calling the method.')
        if opt_aux_num is None and data.meta_data['store_full_solution']:
            opt_aux_num = data.opt_aux(data._opt_aux_num[t_ind])
        elif opt_aux_num is not None:
            pass
        else:
            raise Exception('Cannot plot predictions if full solution is not stored or supplied (opt_aux_num) when calling the method.')

        opt_p_num = data.opt_p(data.opt_p_num[t_ind])

        # Plot predictions:
        self.reset_prop_cycle()
        lines = []
        for line_i in self.line_list:
            line_i['predkwargs'].update(pltkwargs)
            # Fix color for the robust trajectories according to the current state of the cycler.
            if 'color' not in line_i['predkwargs']:
                color = next(line_i['ax']._get_lines.prop_cycler)['color']
                line_i['predkwargs'].update({'color':color})


            # Choose time array depending on variable type (states with n+1 steps)
            if line_i['var_type'] in ['_x', '_z']:
                time = t_now + np.arange(n_horizon+1)*t_step
            else:
                time = t_now + np.arange(n_horizon)*t_step

            # Plot states etc. as continous quantities and inputs as steps.
            if line_i['var_type'] in ['_x', '_z']:
                # Loop over all element of the variable (in case it is a vector)
                for i in range(data.model[line_i['var_type']][line_i['var_name']].shape[0]):
                    # pred is a n_horizon x n_branches array.
                    pred = vertcat(*opt_x_num[line_i['var_type'],:,lambda v: horzcat(*v),:, -1, line_i['var_name'], i])
                    # sort pred such that each column belongs to one scenario
                    pred = pred.full()[range(pred.shape[0]),structure_scenario.T].T
                    lines.extend(line_i['ax'].plot(time, pred, **line_i['predkwargs']))
            elif line_i['var_type'] in ['_u']:
                for i in range(data.model[line_i['var_type']][line_i['var_name']].shape[0]):
                    pred = vertcat(*opt_x_num[line_i['var_type'],:,lambda v: horzcat(*v),:,line_i['var_name'], i])
                    pred = pred.full()[range(pred.shape[0]),structure_scenario[:-1,:].T].T
                    lines.extend(line_i['ax'].step(time, pred, **line_i['predkwargs']))
            elif line_i['var_type'] in ['_aux']:
                for i in range(data.model[line_i['var_type']][line_i['var_name']].shape[0]):
                    pred = vertcat(*opt_aux_num['_aux',:,lambda v: horzcat(*v),:,line_i['var_name'], i])
                    pred = pred.full()[range(pred.shape[0]),structure_scenario[:-1,:].T].T
                    lines.extend(line_i['ax'].plot(time, pred, **line_i['predkwargs']))
            elif line_i['var_type'] in ['_tvp']:
                for i in range(data.model[line_i['var_type']][line_i['var_name']].shape[0]):
                    pred = vertcat(*opt_p_num['_tvp',:, line_i['var_name'], i]).full()
                    lines.extend(line_i['ax'].plot(time, pred, **line_i['predkwargs']))


        return lines

def default_plot(model, states_list=None, inputs_list=None, aux_list=None, **kwargs):
    """Pass a :py:class:`do_mpc.model.Model` object and create a default **do-mpc** plot.
    By default all states, inputs and auxiliary expressions are plotted on individual axes.
    Pass lists of states, inputs and aux names (string) to plot only a subset of these
    trajectories.

    Returns a figure, axis and configured :py:class:`Graphics` object.

    :param model: Configured model that contains all information about states, inputs etc.
    :type model: :py:class:`do_mpc.model.Model`

    :param states_list: List of strings containing a subset of state names defined in py:class:`do_mpc.model.Model`. These states are plotted.
    :type states_list: list

    :param inputs_list: List of strings containing a subset of input names defined in py:class:`do_mpc.model.Model`. These inputs are plotted.
    :type inputs_list: list

    :param aux_list: List of strings containing a subset of auxiliary expression names defined in py:class:`do_mpc.model.Model`. These values are plotted.
    :type aux_list: list

    :param kwargs: Further arguments are passed to the call of ``plt.subplots(n_plot, 1, sharex=True, **kwargs)``.
    :type kwargs:


    :return:
        * fig *(Matplotlib figure)*
        * ax *(Matplotlib axes)*
        * configured :py:class:`Graphics` object (Graphics)

    """
    assert model.flags['setup'] == True, 'Model must be setup. Please call model.setup() first.'

    err_message = '{} contains invalid keys. Must be a subset of {}. You have {}.'
    if states_list is None:
        states_list = model._x.keys()
    else:
        assert set(states_list).issubset(model._x.keys()), err_message.format('states_list',model._x.keys(), states_list)

    if inputs_list is None:
        inputs_list = model._u.keys()
        # Pop default variable:
        inputs_list.pop(0)
    else:
        assert set(inputs_list).issubset(model._u.keys()), err_message.format('inputs_list',model._u.keys(), inputs_list)

    if aux_list is None:
        aux_list = model._aux.keys()
        # Pop default variable:
        aux_list.pop(0)
    else:
        assert set(aux_list).issubset(model._aux.keys()), err_message.format('aux_list',model._aux.keys(), aux_list)

    n_x = len(states_list)
    n_u = len(inputs_list)
    n_aux = len(aux_list)

    n_plot = n_x + n_u + n_aux

    # Create figure:
    fig, ax = plt.subplots(n_plot, 1, sharex=True, **kwargs)

    # Catch special cases:
    if n_plot == 0:
        raise Exception('Nothing to plot.')
    elif n_plot == 1:
        ax = [ax]

    # Create graphics instance:
    graphics = Graphics()

    # Add lines/ labels for states:
    for i, x_i in enumerate(states_list):
        graphics.add_line('_x', x_i, ax[i])
        ax[i].set_ylabel(x_i)
    # Add lines/ labels for inputs:
    for i, u_i in enumerate(inputs_list, n_x):
        graphics.add_line('_u', u_i, ax[i])
        ax[i].set_ylabel(u_i)
    # Add lines/ labels for auxiliary expressions:
    for i, aux_i in enumerate(aux_list, n_x+n_u):
        graphics.add_line('_aux', aux_i, ax[i])
        ax[i].set_ylabel(aux_i)

    ax[-1].set_xlabel('time')

    fig.align_ylabels()
    fig.tight_layout()

    return fig, ax, graphics
