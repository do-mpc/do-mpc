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

"""
Visualization tools for do-mpc.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as maxes
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
import pdb
import os
from do_mpc.tools import IndexedProperty, Structure
import do_mpc
from typing import Union,Tuple

# Define what is included in the Sphinx documentation.
__all__ = ['Graphics', 'default_plot', 'animate']

class Graphics:
    """Graphics module to present the results of **do-mpc**.
    The module is independent of all other modules and can be used optionally.
    The module can also be used with pickled result files in post-processing for flexible and custom graphics.

    The graphics module is based on Matplotlib and allows for fully customizable, publication ready graphics and animations.

    The Graphics module is initialized with an :py:class:`do_mpc.data.Data` or :py:class:`do_mpc.data.MPCData`
    module and will showcase this data.

    User defined graphics are configured prior to plotting results, e.g.:

    ::

        mpc = do_mpc.controller.MPC(model)
        ...

        # Initialize graphic:
        graphics = do_mpc.graphics.Graphics(mpc.data)

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
    The method created (empty) line objects for each plotted variable.
    The graphic is updated with the most recent data with :py:func:`Graphics.plot_results`.
    Furthermore, the module contains the :py:func:`Graphics.plot_predictions` method which is applicable only for :py:class:`do_mpc.data.MPCData`,
    and can be used to show the predicted trajectories.

    Note:
        A high-level API for obtaining a configured :py:class:`Graphics` module is the :py:func:`default_plot` function.
        Use this function and the obtained :py:class:`Graphics` module in the developement process.

    Animations can be setup with the follwing loop:

    ::

        for k in range(50):
            u0 = mpc.make_step(x0)
            y_next = simulator.make_step(u0)
            x0 = estimator.make_step(y_next)

            graphics.plot_results()
            graphics.plot_predictions()
            graphics.reset_axes()
            plt.show()
            plt.pause(0.01)

    Args:
        data: Data object from the **do-mpc** modules (simulator, estimator, controller)
    """
    def __init__(self, data:Union[do_mpc.data.Data,do_mpc.data.MPCData]):
        self.line_list = []
        self.ax_list  = []
        self.color = plt.rcParams['axes.prop_cycle'].by_key()['color']

        self.data = data

        self._result_lines = Structure()


        self._pred_lines = Structure()

    @property
    def result_lines(self, powerind:tuple=None)->list:
        """Structure that holds the result line objects.
        Query this structure with power indices.
        The power indices must have the following order:

        ::

            result_lines[var_type, var_name, i]

        where

        * ``var_type`` refers to ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``, ``_aux``

        * ``var_name`` refers to the user-defined names in the :py:class:`do_mpc.model.Model`

        * Index ``i`` is applicable if the selecte variable is vector valued.

        Note that (e.g.) ``result_lines['_x']`` will return all lines for all states and
        ``result_lines.full`` can be used to retrieve all line objects.

        This property can be used to query and configure specific lines in the current graphic.

        **Example:**

        ::

            # Update properties for all lines:
            for line_i in graphics.result_lines.full:
                line_i.set_linewidth(2)
                line_i.set_alpha(0.5)

        An extensive list of all line properties can be found `here <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.lines.Line2D.html>`_.

        Args:
            powerind: Tuple of indices (power indices) to obtain the desired line obects

        Returns:
            List of line objects.
        """
        # Note this property is a wrapper to showcase the documentation.
        return self._result_lines

    @property
    def pred_lines(self, powerind:tuple=None)->list:
        """Structure that holds the prediction line objects.
        Query this structure with power indices.
        The power indices must have the following order:

        ::

            pred_lines[var_type, var_name, i, k]

        where

        * ``var_type`` refers to ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``, ``_aux``

        * ``var_name`` refers to the user-defined names in the :py:class:`do_mpc.model.Model`

        * Use ``i`` to index vector valued variables (choose 0 for scalars).

        * Use ``k`` to select the k-th scenario (for robust MPC). Note the ``k=0`` is the nominal case.

        Note that (e.g.) ``pred_lines['_x']`` will return all lines for all states and
        ``pred_lines.full`` can be used to retrieve all line objects.

        This property can be used to query and configure specific lines in the current graphic.

        **Example:**

        ::

            # Update properties for all lines:
            for line_i in graphics.pred_lines.full:
                line_i.set_linewidth(2)
                line_i.set_alpha(0.5)

        An extensive list of all line properties can be found `here <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.lines.Line2D.html>`_.

        Args:
            powerind: Tuple of indices (power indices) to obtain the desired line obects

        Returns:
            List of line objects.
        """
        # Note this property is a wrapper to showcase the documentation.
        return self._pred_lines

    def reset_axes(self)->None:
        """Relimits and scales all axes.
        This method calls

        ::

            ax.relim()
            ax.autoscale()

        on all axes instances in the class.
        """
        for ax_i in self.ax_list:
            ax_i.relim()
            ax_i.autoscale()

    def reset_prop_cycle(self)->None:
        """Resets the property cycle for all axes which were passed with :py:func:`Graphics.add_line`.
        The matplotlib color cycler is restarted.
        """
        for ax_i in self.ax_list:
            ax_i.set_prop_cycle(None)

    def clear(self, lines:list=None)->None:
        """Clears all data from lines.

        """
        if lines==None:
            for line_i in self.result_lines.master:
                line_i.set_data([],[])
        else:
            assert isinstance(lines, list), 'lines must be of type list.'
            for line_i in lines:
                line_i.set_data([],[])

    def add_line(self, var_type:str, var_name:str, axis:maxes.Axes, **pltkwargs)->None:
        """``add_line`` is called during setting up the :py:class:`Graphics` class. This is typically the last step of configuring **do-mpc**.
        Each call of :py:func:`Graphics.add_line` adds a line to the passed axis according to the variable type
        (``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``, ``_aux``)
        and its name (as defined in the :py:class:`do_mpc.model.Model`).
        Furthermore, all valid matplotlib .plot arguments can be passed as optional keyword arguments,
        e.g.: ``linewidth``, ``color``, ``alpha``.

        Note:
            Lines can also be configured after adding them with this method.
            Use the :py:func:`result_lines` and :py:func:`pred_lines` attributes for this purpose.

        Args:
            var_type: Variable type to be plotted. Valid arguments are ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``, ``_aux``.
            var_name: Variable name. Must reference the names defined in the model for the given variable type.
            axis: Axis object on which to plot the line(s).
            pltkwargs: Valid matplotlib pyplot keyword arguments (e.g.: ``linewidth``, ``color``, ``alpha``)

        Raises:
            assertion: var_type argument must be a string
            assertion: var_name argument must be a string
            assertion: var_type argument must reference to the valid var_types of do-mpc models.
            assertion: axis argument must be matplotlib axes object.
        """
        assert isinstance(var_type, str), 'var_type argument must be a string. You have: {}'.format(type(var_type))
        assert isinstance(var_name, str), 'var_name argument must be a string. You have: {}'.format(type(var_name))
        assert var_type in ['_x', '_u', '_z', '_tvp', '_p', '_aux'], 'var_type argument must reference to the valid var_types of do-mpc models. Note that _aux_expression are currently not supported for plotting.'
        assert isinstance(axis, maxes.Axes), 'axis argument must be matplotlib axes object.'

        if var_type == '_u':
            pltkwargs.update(drawstyle='steps-post')

        self.result_lines[var_type, var_name] = axis.plot(self.data['_time'] , self.data[var_type, var_name], **pltkwargs)

        if self.data.dtype == 'MPC' and self.data.meta_data['store_full_solution']:
            # y_data has shape (n_elem, n_horizon, n_scenario), where n_elem = 1 for scalars and >1 for vectors
            y_data = self.data.prediction((var_type, var_name))
            x_data = np.zeros(y_data.shape[1])
            for i in range(y_data.shape[0]):
                # Loop is only meaningful is variable is a vector.
                color = self.result_lines[var_type, var_name][i].get_color()
                # Default values:
                pltkwargs.update(color=color, linestyle='--')
                self.pred_lines[var_type, var_name, i] = axis.plot(x_data, y_data[i], **pltkwargs)

        self.ax_list.append(axis)

    def plot_results(self, t_ind:int=-1)->None:
        """Plots the results stored in the data object.
        Use the ``t_ind`` parameter to plot only until the given time index. This can be used in post-processing for animations.

        Args:
            t_ind: Plot results up until this time index.

        Raises:
            assertion: t_ind argument must be a int
            assertion: t_ind argument must not exceed the length of the results
        """
        assert isinstance(t_ind, int), 't_ind argument must be of type integer.'
        n_elem = self.data['_time'].shape[0]
        assert abs(t_ind) <= n_elem, 't_ind={} argument is out of range for recorded data with {} elements.'.format(t_ind, n_elem)

        for line_i, ind_i in zip(self.result_lines.master, self.result_lines.powerindex):
            # ind_i will look something like: ('_x', 'Temperature', 0) and is a tuple.
            if t_ind == -1:
                # Non-inclusive indexing: Last element is missing due to slice ...
                line_i.set_data(self.data['_time'] , self.data[ind_i])
            else:
                line_i.set_data(self.data['_time'][:t_ind+1] , self.data[ind_i][:t_ind+1])

    def plot_predictions(self, t_ind:int=-1)->None:
        """Plots the predicted trajectories for the plot configuration.
        The predicted trajectories are part of the optimal solution at each timestep
        and are **optionally** stored in the :py:class:`do_mpc.data.MPCData` object.

        Warnings:
            This method requires that the optimal solution is stored in the :py:class:`do_mpc.data.MPCData` instance.
            Storing the optimal solution must be activated with :py:func:`do_mpc.controller.MPC.set_param`.

        The ``plot_predictions`` method can only be called with data from the :py:class:`do_mpc.controller.MPC` object
        and raises an error if called with data from other objects.
        Use the ``t_ind`` parameter to plot the prediction for the given time instance.
        This can be used in post-processing for animations.

        Args:
            t_ind: Plot predictions at this time index.

        Raises:
            assertion: Can only call plot_predictions with data object from do-mpc optimizer
            Exception: Cannot plot predictions if full solution is not stored or supplied when calling the method
            assertion: t_ind argument must be a int
            assertion: t_ind argument must not exceed the length of the results
        """
        assert self.data.dtype == 'MPC', 'Plotting predictions is only possible for MPC data.'
        assert self.data.meta_data['store_full_solution'], 'Optimal trajectory is not stored. Please update your MPC settings.'
        assert isinstance(t_ind, int), 't_ind argument must be of type integer.'
        n_elem = self.data['_time'].shape[0]
        assert abs(t_ind) <= n_elem, 't_ind={} argument is out of range for recorded data with {} elements.'.format(t_ind, n_elem)

        t_now = self.data._time[t_ind]
        t_step = self.data.meta_data['t_step']

        for line_i, ind_i in zip(self.pred_lines.master, self.pred_lines.powerindex):

            y_data = self.data.prediction(ind_i[:-1], t_ind=t_ind)[0, :,ind_i[-1]]
            x_data = t_now + np.arange(y_data.shape[0])*t_step
            line_i.set_data(x_data , y_data)

def default_plot(data, 
                 states_list:list=None, 
                 dae_states_list:list=None, 
                 inputs_list:list=None, 
                 aux_list:list=None, 
                 **kwargs)->Tuple[plt.figure,plt.axes,Graphics]:
    """Pass a :py:class:`do_mpc.data.Data` object and create a default **do-mpc** plot.
    By default all states, inputs and auxiliary expressions are plotted on individual axes.
    Pass lists of states, inputs and aux names (string) to plot only a subset of these
    trajectories.

    Returns a figure, axis and configured :py:class:`Graphics` object.

    Args:
        data: **do-mpc** data instance.
        states_list: List of strings containing a subset of state names defined in py:class:`do_mpc.model.Model`. These states are plotted.
        dae_states_list: List of strings containing a subset of dae states (_z) names defined in py:class:`do_mpc.model.Model`. These states are plotted.
        inputs_list: List of strings containing a subset of input names defined in py:class:`do_mpc.model.Model`. These inputs are plotted.
        aux_list: List of strings containing a subset of auxiliary expression names defined in py:class:`do_mpc.model.Model`. These values are plotted.
        kwargs: Further arguments are passed to the call of ``plt.subplots(n_plot, 1, sharex=True, **kwargs)``.

    Returns:
        Matplotlib ``fig`` and ``ax`` and configured :py:class:`Graphics` object.
    """
    err_message = '{} contains invalid keys. Must be a subset of {}. You have {}.'
    if states_list is None:
        states_list = data.model['_x'].keys()
    else:
        assert set(states_list).issubset(data.model['_x'].keys()), err_message.format('states_list',data.model['_x'].keys(), states_list)

    if dae_states_list is None:
        dae_states_list = data.model['_z'].keys()
        # Pop default variable:
        dae_states_list.pop(0)
    else:
        assert set(dae_states_list).issubset(data.model['_u'].keys()), err_message.format('dae_states_list',data.model['_z'].keys(), dae_states_list)

    if inputs_list is None:
        inputs_list = data.model['_u'].keys()
        # Pop default variable:
        inputs_list.pop(0)
    else:
        assert set(inputs_list).issubset(data.model['_u'].keys()), err_message.format('inputs_list',data.model['_u'].keys(), inputs_list)

    if aux_list is None:
        aux_list = data.model['_aux'].keys()
        # Pop default variable:
        aux_list.pop(0)
    else:
        assert set(aux_list).issubset(data.model['_aux'].keys()), err_message.format('aux_list',data.model['_aux'].keys(), aux_list)

    n_x = len(states_list)
    n_u = len(inputs_list)
    n_aux = len(aux_list)
    n_z = len(dae_states_list)

    n_plot = n_x + n_u + n_aux + n_z

    # Create figure:
    fig, ax = plt.subplots(n_plot, 1, sharex=True, **kwargs)

    # Catch special cases:
    if n_plot == 0:
        raise Exception('Nothing to plot.')
    elif n_plot == 1:
        ax = [ax]

    # Create graphics instance:
    graphics = Graphics(data)

    counter = 0
    # Add lines/ labels for states:
    for i, x_i in enumerate(states_list, counter):
        graphics.add_line('_x', x_i, ax[i])
        ax[i].set_ylabel(x_i)
    counter += n_x
    for i, z_i in enumerate(dae_states_list, counter):
        graphics.add_line('_z', z_i, ax[i])
        ax[i].set_ylabel(z_i)
    counter += n_z
    # Add lines/ labels for inputs:
    for i, u_i in enumerate(inputs_list, counter):
        graphics.add_line('_u', u_i, ax[i])
        ax[i].set_ylabel(u_i)
    counter += n_u
    # Add lines/ labels for auxiliary expressions:
    for i, aux_i in enumerate(aux_list, counter):
        graphics.add_line('_aux', aux_i, ax[i])
        ax[i].set_ylabel(aux_i)

    ax[-1].set_xlabel('time')

    fig.align_ylabels()
    fig.tight_layout()

    return fig, ax, graphics

def animate(graphics:Graphics, 
            fig:plt.figure, 
            n_steps:int=None, 
            export_path:str='./', 
            export_name:str='animation', 
            overwrite:bool=False, 
            format:str='gif', 
            fps:int=5, 
            writer:Union[FFMpegWriter,ImageMagickWriter]=None)->None:
    """Animation helper function.

    Call this function with a configured :py:class:`Graphics` instance and the respective figure.
    This function will export an animation with the results from the :py:class:`do_mpc.data.Data` object.

    Either specify ``format`` and ``fps`` or supply a configured writer (e.g. ``ImageMagickWriter`` for gifs).

    Args:
        graphics: Configured :py:class:`Graphics` instance.
        fig: Matplotlib Figure.
        n_steps: number of time steps for the animation.
        export_path: Path where to export the animation. Directory will be created if it doesn't exist.
        export_name: Name of the resulting animation (gif/mp4) file.
        overwrite: Check if export_name already exists in the supplied directory and overwrite or alter export_name.
        format: Choose between gif or mp4.
        fps: Frames per second for the resulting animation.
        writer: If supplied, the ``fps`` and ``format`` argument are discarded. Use this to configure your own writer.
    """
    if n_steps==None:
        n_steps = graphics.data['_time'].shape[0]

    def update(t_ind):
        print('Writing frame: {} of {}.'.format(t_ind, n_steps))
        graphics.plot_results(t_ind=t_ind)
        graphics.plot_predictions(t_ind=t_ind)
        graphics.reset_axes()
        lines = graphics.result_lines.full+graphics.pred_lines.full
        return lines

    anim = FuncAnimation(fig, update, frames=n_steps, blit=True)

    if writer==None:
        if 'mp4' in format:
            writer = FFMpegWriter(fps=fps, extra_args=['-vcodec', 'libx264'])
            extension='mp4'

        elif 'gif' in format:
            writer = ImageMagickWriter(fps=fps)
            extension='gif'
        else:
            raise Exception('Invalid output format {}. Please choose mp4 or gif.'.format(format))
    else:
        extension=''

    if not os.path.exists(export_path):
        os.makedirs(export_path)
    # Dynamically generate new result name if name is already taken in result_path.
    if overwrite==False:
        ind = 1
        ext_export_name = export_name
        while os.path.isfile(export_path+ext_export_name+'.pkl'):
            ext_export_name = '{ind:03d}_{name}'.format(ind=ind, name=export_name)
            ind += 1
        export_name = ext_export_name

    anim.save('{}{}.{}'.format(export_path, export_name, extension), writer=writer)