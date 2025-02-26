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

# Imports
import warnings
import numpy as np
import do_mpc
from pathlib import Path

import pandas as pd

# import time
from timeit import default_timer as timer
import pickle as pkl
import casadi as ca
from ._ampcsettings import SamplerSettings


class Sampler:
    """Class to sample data for the ApproxMPC.

    This class rnadomly samples the MPC to generate data. This data can furthur be used to train the ApproxMPC.

    .. versionadded:: >v4.5.

    Configuring and setting up the Sampler involves the following steps:

    1. Configure the Sampler controller with :py:class:`SamplerSettings`. The Sampler instance has the attribute ``settings`` which is an instance of :py:class:`SamplerSettings`.

    2. To finalize the class configuration :py:meth:`setup` may be called. This method sets up the MPC and Simulator, if needed.

    **Usage:**

    The Sampler can be used in a closed loop setting by calling the :py:meth:`default_sampling` method. This method starts generating data based on the user settings.

    **Example:**

    .. code-block:: python

        # pushing to class
        mpc.x0 = x0
        mpc.set_initial_guess()

        # approximate mpc
        approx_mpc = ApproxMPC(mpc)
        approx_mpc.settings.n_hidden_layers = 3
        approx_mpc.settings.n_neurons = 50
        approx_mpc.setup()

        # sampler
        sampler = Sampler(mpc)
        sampler.settings.closed_loop_flag = True
        sampler.settings.trajectory_length = 5
        sampler.settings.n_samples = 100
        sampler.setup()
        sampler.default_sampling()


    Args:
        mpc (do_mpc.controller.MPC): The MPC class which is sampled.
    """

    def __init__(self, mpc):
        # storage
        self.mpc = mpc

        # settings
        self._settings = SamplerSettings()

        # init simulator
        self.simulator = do_mpc.simulator.Simulator(self.mpc.model)

        # flags
        self.flags = {
            "setup": False,
        }
        return None

    def setup(self):
        """Setup the Sampler class.

        This method sets up the MPC, and the Simulator if needed.

        .. note::
            This method should be called before calling any other method.

        Returns
        -------
        None
        """

        assert self.flags["setup"] is False, "Setup can only be once."
        self.flags.update(
            {
                "setup": True,
            }
        )

        # mandatory check for sanity
        self._settings.check_for_mandatory_settings()

        # init
        self.lbx = ca.DM(self.mpc._x_lb).full()
        self.ubx = ca.DM(self.mpc._x_ub).full()
        self.lbu = ca.DM(self.mpc._u_lb).full()
        self.ubu = ca.DM(self.mpc._u_ub).full()

        # extra setup for closed loop
        if self.settings.closed_loop_flag:
            self.setup_simulator()
            self.estimator = do_mpc.estimator.StateFeedback(model=self.mpc.model)

        # end of setup
        return None

    @property
    def settings(self):
        """Sampler settings.

        This attribute is an instance of :py:class:`SamplerSettings`. It is used to configure the Sampler.

        Returns
        -------
        SamplerSettings
            The settings for the Sampler.
        """
        return self._settings

    @settings.setter
    def settings(self, val):
        """Sampler settings.

        This attribute is an instance of :py:class:`SamplerSettings`. It is used to configure the Sampler.

        Raises
        ------
        UserWarning
            Cannot change the settings attribute.
        """
        warnings.warn("Cannot change the settings attribute")

    @property
    def simulator_settings(self):
        """Simulator settings.

        This attribute is an instance of :py:class:`do_mpc.simulator.SimulatorSettings`. It is used to configure the Simulator.

        Returns
        -------
        do_mpc.simulator.SimulatorSettings
            The settings for the Simulator.
        """
        return self.simulator.settings

    @simulator_settings.setter
    def simulator_settings(self, val):
        """Simulator settings.

        This attribute is an instance of :py:class:`do_mpc.simulator.SimulatorSettings`. It is used to configure the Simulator.

        Raises
        ------
        UserWarning
            Cannot change the simulatro settings attribute.
        """
        warnings.warn("Cannot change the simulatro settings attribute")

    def setup_simulator(self):
        """Setup the Simulator.

        This function is only used when the `closed_loop_flag` is set to True. It sets up the Simulator for the closed loop sampling.

        Returns
        -------
        None
        """
        # assert to prevent robust mpc from executing
        assert self.mpc.settings.n_robust == 0, (
            "Sampler with robust mpc not implemented yet."
        )

        # extracting t_step from mpc
        self.simulator.settings.t_step = self.mpc.settings.t_step

        # extracting tvp from the mpc class
        self.simulator.set_tvp_fun(self.mpc.tvp_fun)

        # extracting p from the mpc class
        self.simulator.set_p_fun(self.mpc.p_fun)

        # simulator setup
        self.simulator.setup()

        # end
        return None

    def default_sampling(self):
        """Default sampling method.

        This method is used to sample data for the ApproxMPC. The method generates data based on the user settings.

        Returns
        -------
        None
        """

        self.approx_mpc_sampling_plan_box()

        if self.settings.closed_loop_flag:
            self.approx_mpc_closed_loop_sampling()
        else:
            self.approx_mpc_open_loop_sampling()

        return None

    def approx_mpc_sampling_plan_box(self):
        """Generate sampling plan for the ApproxMPC.

        This method generates a sampling plan for the ApproxMPC. The sampling plan is saved in the data directory.

        Returns
        -------
        None
        """

        overwrite = self.settings.overwrite_sampler

        # Samples
        data_dir = Path(self.settings.data_dir)
        sampling_plan_name = "sampling_plan" + "_n" + str(self.settings.n_samples)
        id_precision = np.ceil(np.log10(self.settings.n_samples)).astype(int)

        def gen_x0():
            x0 = np.random.uniform(self.lbx, self.ubx)
            return x0

        def gen_u_prev():
            u_prev = np.random.uniform(self.lbu, self.ubu)
            return u_prev

        assert self.settings.n_samples <= 10 ** (id_precision + 1), (
            "Not enough ID-digits to save samples"
        )

        # Initialize sampling planner
        sp = do_mpc.sampling.SamplingPlanner()
        sp.set_param(overwrite=overwrite)
        sp.set_param(id_precision=id_precision)
        sp.data_dir = data_dir.__str__() + "/"

        # Set sampling vars
        sp.set_sampling_var("x0", gen_x0)
        sp.set_sampling_var("u_prev", gen_u_prev)

        # Generate sampling plan
        plan = sp.gen_sampling_plan(n_samples=self.settings.n_samples)

        # Export
        sp.export(sampling_plan_name)

        # end of fucntion
        return None

    def approx_mpc_open_loop_sampling(self):
        """Open loop sampling for the ApproxMPC.

        This method generates open loop samples for the ApproxMPC. The samples are saved in the data directory.

        Returns
        -------
        None
        """
        overwrite_sampler = self.settings.overwrite_sampler

        n_samples = self.settings.n_samples
        mpc = self.mpc

        suffix = "_n" + str(n_samples)
        sampling_plan_name = "sampling_plan"
        sample_name = "sample"
        data_dir = Path(self.settings.data_dir)

        # How are samples named? (DEFAULT)
        sampling_plan_name = sampling_plan_name + suffix  # 'sampling_plan'+suffix

        samples_dir = data_dir.joinpath("samples" + suffix)

        # Data
        data_file_name = "data"

        # Sampling functions
        def run_mpc_one_step(x0, u_prev):
            mpc.reset_history()
            mpc.x0 = x0
            mpc.u0 = u_prev
            mpc.set_initial_guess()

            start = timer()
            u0 = mpc.make_step(x0)
            end = timer()

            stats = {}
            stats["t_make_step"] = end - start
            stats["success"] = mpc.solver_stats["success"]
            stats["iter_count"] = mpc.solver_stats["iter_count"]

            if "t_wall_total" in mpc.solver_stats:
                stats["t_wall_total"] = mpc.solver_stats["t_wall_total"]
            else:
                stats["t_wall_total"] = np.nan

                return u0, stats

        # Sampling function
        def sample_function(x0, u_prev):
            return run_mpc_one_step(x0, u_prev)

        # Import sampling plan
        with open(data_dir.joinpath(sampling_plan_name + ".pkl"), "rb") as f:
            plan = pkl.load(f)

        sampler = do_mpc.sampling.Sampler(plan)
        sampler.data_dir = str(samples_dir) + "/"
        sampler.set_param(overwrite=overwrite_sampler)
        sampler.set_param(sample_name=sample_name)

        sampler.set_sample_function(sample_function)

        sampler.sample_data()

        # Data Handling
        dh = do_mpc.sampling.DataHandler(plan)

        dh.data_dir = str(samples_dir) + "/"
        dh.set_param(sample_name=sample_name)
        dh.set_post_processing("u0", lambda x: x[0])
        dh.set_post_processing("status", lambda x: x[1]["success"])
        dh.set_post_processing("t_make_step", lambda x: x[1]["t_make_step"])
        dh.set_post_processing("t_wall", lambda x: x[1]["t_wall_total"])
        dh.set_post_processing("iter_count", lambda x: x[1]["iter_count"])

        df = pd.DataFrame(dh[:])
        n_data = df.shape[0]
        df.to_pickle(
            str(data_dir)
            + "/"
            + data_file_name
            + "_n{}".format(n_data)
            + "_all"
            + ".pkl"
        )

        # Save
        df = pd.DataFrame(dh.filter(output_filter=lambda status: status == True))
        n_data_opt = df.shape[0]
        df.to_pickle(
            str(data_dir)
            + "/"
            + data_file_name
            + "_n{}".format(n_data)
            + "_opt"
            + ".pkl"
        )

        # Save all
        return None

    def approx_mpc_closed_loop_sampling(self):
        """Closed loop sampling for the ApproxMPC.

        This method generates closed loop samples for the ApproxMPC. The samples are saved in the data directory.

        Returns
        -------
        None
        """

        n_samples = self.settings.n_samples
        mpc = self.mpc
        trajectory_length = self.settings.trajectory_length
        overwrite_sampler = self.settings.overwrite_sampler

        suffix = "_n" + str(n_samples)
        sampling_plan_name = "sampling_plan"
        sample_name = "sample"
        data_dir = Path(self.settings.data_dir)

        sampling_plan_name = sampling_plan_name + suffix  # 'sampling_plan'+suffix

        # overwrite_sampler = False
        samples_dir = data_dir.joinpath("samples" + suffix)
        # samples_dir = data_dir+'samples' + suffix

        # Data
        data_file_name = "data"

        # Sampling functions
        def run_mpc_closed_loop(x0, u_prev):
            mpc.reset_history()
            mpc.x0 = x0
            mpc.u0 = u_prev
            u_prev_total = np.zeros((trajectory_length, self.mpc.model.n_u))
            mpc.set_initial_guess()

            start = timer()
            mpc.reset_history()
            self.simulator.reset_history()
            self.estimator.reset_history()

            mpc.x0 = x0
            self.simulator.x0 = x0
            self.estimator.x0 = x0

            mpc.set_initial_guess()
            u_prev_curr = u_prev

            # run the closed loop for 150 steps
            for k in range(trajectory_length):
                u_prev_total[k] = u_prev_curr.reshape((self.mpc.model.n_u,))
                u0 = mpc.make_step(x0)
                u_prev_curr = u0
                if mpc.solver_stats["success"] == False:
                    break
                y_next = self.simulator.make_step(u0)
                x0 = self.estimator.make_step(y_next)

            end = timer()

            stats = {}
            stats["t_make_step"] = end - start
            stats["success"] = mpc.solver_stats["success"]
            stats["iter_count"] = mpc.solver_stats["iter_count"]

            if "t_wall_total" in mpc.solver_stats:
                stats["t_wall_total"] = mpc.solver_stats["t_wall_total"]
            else:
                stats["t_wall_total"] = np.nan

            return self.simulator.data, stats, u_prev_total

        def sample_function(x0, u_prev):
            return run_mpc_closed_loop(x0, u_prev)

        # Sampling Plan
        with open(data_dir.joinpath(sampling_plan_name + ".pkl"), "rb") as f:
            plan = pkl.load(f)

        # Sampler
        sampler = do_mpc.sampling.Sampler(plan)
        sampler.data_dir = str(samples_dir) + "/"
        sampler.set_param(overwrite=overwrite_sampler)
        sampler.set_param(sample_name=sample_name)

        sampler.set_sample_function(sample_function)

        # Main - Sample Data
        sampler.sample_data()

        # Data Handling
        dh = do_mpc.sampling.DataHandler(plan)

        dh.data_dir = str(samples_dir) + "/"
        dh.set_param(sample_name=sample_name)
        dh.set_post_processing("u0", lambda x: x[0]["_u"])
        dh.set_post_processing("x0", lambda x: x[0]["_x"])
        dh.set_post_processing("u_prev", lambda x: x[2])
        dh.set_post_processing("status", lambda x: x[1]["success"])
        dh.set_post_processing("t_make_step", lambda x: x[1]["t_make_step"])
        dh.set_post_processing("t_wall", lambda x: x[1]["t_wall_total"])
        dh.set_post_processing("iter_count", lambda x: x[1]["iter_count"])
        df = pd.DataFrame(dh[:])
        n_data = df.shape[0]
        df.to_pickle(
            str(data_dir)
            + "/"
            + data_file_name
            + "_n{}".format(n_data)
            + "_all"
            + ".pkl"
        )

        # Save
        df = pd.DataFrame(dh.filter(output_filter=lambda status: status == True))
        n_data_opt = df.shape[0]
        df.to_pickle(
            str(data_dir)
            + "/"
            + data_file_name
            + "_n{}".format(n_data)
            + "_opt"
            + ".pkl"
        )

        # end of function
        return None
