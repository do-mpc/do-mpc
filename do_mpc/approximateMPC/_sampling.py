
# %% Imports
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
# %% Config
#####################################################
class Sampler:
    def __init__(self, appx_mpc):

        # storage
        self.appx_mpc = appx_mpc

        # settings
        self._settings = SamplerSettings()

        # init simulator
        self.simulator = do_mpc.simulator.Simulator(self.appx_mpc.mpc.model)

        # flags
        self.flags = {
            'setup': False,
        }
        return None

    def setup(self):
        assert self.flags['setup'] is False, "Setup can only be once."
        self.flags.update({
            'setup': True,
        })

        # mandatory check for sanity
        self._settings.check_for_mandatory_settings()

        # init
        self.lbx = ca.DM(self.appx_mpc.mpc._x_lb).full()
        self.ubx = ca.DM(self.appx_mpc.mpc._x_ub).full()
        self.lbu = ca.DM(self.appx_mpc.mpc._u_lb).full()
        self.ubu = ca.DM(self.appx_mpc.mpc._u_ub).full()

        # extra setup for closed loop
        if self.settings.closed_loop_flag:
            self.setup_simulator()
            self.estimator = do_mpc.estimator.StateFeedback(model= self.appx_mpc.mpc.model)

        # end of setup
        return None

    @property
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, val):
        warnings.warn('Cannot change the settings attribute')

    @property
    def simulator_settings(self):
        return self.simulator.settings

    @simulator_settings.setter
    def simulator_settings(self, val):
        warnings.warn('Cannot change the simulatro settings attribute')

    def setup_simulator(self):
        # assert to prevent robust mpc from executing
        assert self.appx_mpc.mpc.settings.n_robust == 0, 'Sampler with robust mpc not implemented yet.'

        # extracting t_step from mpc
        self.simulator.settings.t_step = self.appx_mpc.mpc.settings.t_step

        # extracting tvp from the mpc class
        self.simulator.set_tvp_fun(self.appx_mpc.mpc.tvp_fun)

        # extracting p from the mpc class
        self.simulator.set_p_fun(self.appx_mpc.mpc.p_fun)

        # simulator setup
        self.simulator.setup()

        # end
        return None

    def default_sampling(self):

        self.approx_mpc_sampling_plan_box()

        if self.settings.closed_loop_flag:
            self.approx_mpc_closed_loop_sampling()
        else:
            self.approx_mpc_open_loop_sampling()



    def boxes_from_mpc(self,mpc):
        pass

    def approx_mpc_sampling_plan_box(self):

        #n_samples = self.settings.n_samples

        overwrite = self.settings.overwrite_sampler

        # Samples
        data_dir=Path(self.settings.data_dir)
        sampling_plan_name = 'sampling_plan' + '_n' + str(self.settings.n_samples)
        #overwrite = True
        id_precision = np.ceil(np.log10(self.settings.n_samples)).astype(int)


        #####################################################

        # %% Functions

        def gen_x0():
            x0 = np.random.uniform(self.lbx, self.ubx)
            return x0

        def gen_u_prev():
            u_prev = np.random.uniform(self.lbu, self.ubu)
            return u_prev

        # %%
        # Sampling Plan

        assert self.settings.n_samples <= 10 ** (id_precision + 1), "Not enough ID-digits to save samples"
        # Initialize sampling planner
        sp = do_mpc.sampling.SamplingPlanner()
        sp.set_param(overwrite=overwrite)
        sp.set_param(id_precision=id_precision)
        sp.data_dir = data_dir.__str__()+"/"

        # Set sampling vars
        sp.set_sampling_var('x0', gen_x0)
        sp.set_sampling_var('u_prev', gen_u_prev)

        # Generate sampling plan
        plan = sp.gen_sampling_plan(n_samples=self.settings.n_samples)

        # Export
        sp.export(sampling_plan_name)

        # end of fucntion
        return None


    #####################################################
    # %% MPC
    def approx_mpc_open_loop_sampling(self):
        overwrite_sampler = self.settings.overwrite_sampler

        n_samples = self.settings.n_samples
        mpc = self.appx_mpc.mpc

        # %% Config
        #####################################################
        suffix='_n'+str(n_samples)
        sampling_plan_name = 'sampling_plan'
        sample_name = 'sample'
        data_dir=Path(self.settings.data_dir)
        #return_full_mpc_data = True

        ## How are samples named? (DEFAULT)
        #sample_name = 'sample'
        #suffix = '_n4000'
        sampling_plan_name = sampling_plan_name + suffix  # 'sampling_plan'+suffix

        #overwrite_sampler = False
        samples_dir = data_dir.joinpath('samples' + suffix)
        #samples_dir = data_dir+'samples' + suffix

        # Data
        #test_run = False
        # filter_success_runs = False
        data_file_name = 'data'

        # Assertion for scaling
        #for val in [mpc._x_scaling.cat, mpc._p_scaling.cat, mpc._u_scaling.cat]:
        #    assert (np.array(val)==1).all(), "you have to consider scaling: change opt_x_num to consider scaled values"

        # %% NLP Handler
        # setup NLP Handler
        #nlp_handler = NLPHandler(mpc)

        # %% Functions

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
            stats["t_make_step"] = end-start
            stats["success"] = mpc.solver_stats["success"]
            stats["iter_count"] = mpc.solver_stats["iter_count"]

            if "t_wall_total" in mpc.solver_stats:
                stats["t_wall_total"] = mpc.solver_stats["t_wall_total"]
            else:
                stats["t_wall_total"] = np.nan

            #if return_full:
            #    ### get solution
            #    nlp_sol, p_num = nlp_handler.get_mpc_sol(mpc)
            #    z_num = nlp_handler.extract_numeric_primal_dual_sol(nlp_sol)
                ### reduced solution
                # z_num, p_num = nlp_handler.get_reduced_primal_dual_sol(nlp_sol,p_num)
            #    return u0, stats, np.array(z_num), np.array(p_num), mpc.data
            #else:
                return u0, stats

        # Sampling function
        def sample_function(x0, u_prev):
            return run_mpc_one_step(x0, u_prev)

        # %% Sampling Plan
        # Import sampling plan
        # with open(data_dir+sampling_plan_name+'.pkl','rb') as f:
        with open(data_dir.joinpath(sampling_plan_name+'.pkl'),'rb') as f:
            plan = pkl.load(f)

        # %% Sampler
        sampler = do_mpc.sampling.Sampler(plan)
        sampler.data_dir = str(samples_dir)+'/'
        sampler.set_param(overwrite=overwrite_sampler)
        sampler.set_param(sample_name=sample_name)

        sampler.set_sample_function(sample_function)

        # %% Main - Sample Data
        #if test_run:
        #    sampler.sample_idx(0)
        #else:
        sampler.sample_data()

        # %% Data Handling
        dh = do_mpc.sampling.DataHandler(plan)

        dh.data_dir = str(samples_dir)+'/'
        dh.set_param(sample_name = sample_name)
        dh.set_post_processing('u0', lambda x: x[0])
        dh.set_post_processing('status', lambda x: x[1]["success"])
        dh.set_post_processing('t_make_step', lambda x: x[1]["t_make_step"])
        dh.set_post_processing('t_wall', lambda x: x[1]["t_wall_total"])
        dh.set_post_processing('iter_count', lambda x: x[1]["iter_count"])
        #dh.set_post_processing('n_samples', lambda x: x[1]["iter_count"])
        #if return_full_mpc_data == True:
        #    dh.set_post_processing('z_num', lambda x: x[2])
        #    dh.set_post_processing('p_num', lambda x: x[3])
        #    dh.set_post_processing('mpc_data', lambda x: x[4])

        # if filter_success_runs:
        #     df = pd.DataFrame(dh.filter(output_filter = lambda status: status==True))
        # else:
        #     df = pd.DataFrame(dh[:])

        # n_data = df.shape[0]

        # # %% Save
        # if filter_success_runs:
        #     df.to_pickle(str(data_dir) +'/' + data_file_name + '_n{}'.format(n_data) + '_opt' + '.pkl')
        # else:
        #     df.to_pickle(str(data_dir) +'/' + data_file_name + '_n{}'.format(n_data) + '_all' + '.pkl')

        df = pd.DataFrame(dh[:])
        n_data = df.shape[0]
        df.to_pickle(str(data_dir) + '/' + data_file_name + '_n{}'.format(n_data) + '_all' + '.pkl')
        # %% Save
        # Filter opt and Save
        df = pd.DataFrame(dh.filter(output_filter = lambda status: status==True))
        n_data_opt = df.shape[0]
        df.to_pickle(str(data_dir) +'/' + data_file_name + '_n{}'.format(n_data) + '_opt' + '.pkl')

        # Save all


    def approx_mpc_closed_loop_sampling(self):
        # %% Config
        #####################################################
        n_samples = self.settings.n_samples
        mpc = self.appx_mpc.mpc
        trajectory_length = self.settings.trajectory_length
        overwrite_sampler = self.settings.overwrite_sampler


        suffix = '_n' + str(n_samples)
        sampling_plan_name = 'sampling_plan'
        sample_name = 'sample'
        data_dir = Path(self.settings.data_dir)
        # return_full_mpc_data = True

        ## How are samples named? (DEFAULT)
        # sample_name = 'sample'
        # suffix = '_n4000'
        sampling_plan_name = sampling_plan_name + suffix  # 'sampling_plan'+suffix

        # overwrite_sampler = False
        samples_dir = data_dir.joinpath('samples' + suffix)
        # samples_dir = data_dir+'samples' + suffix

        # Data
        # test_run = False
        # filter_success_runs = False
        data_file_name = 'data'

        # Assertion for scaling
        # for val in [mpc._x_scaling.cat, mpc._p_scaling.cat, mpc._u_scaling.cat]:
        #    assert (np.array(val)==1).all(), "you have to consider scaling: change opt_x_num to consider scaled values"

        # %% NLP Handler
        # setup NLP Handler
        # nlp_handler = NLPHandler(mpc)

        # %% Functions

        # Sampling functions
        def run_mpc_closed_loop(x0, u_prev):
            mpc.reset_history()
            mpc.x0 = x0
            mpc.u0 = u_prev
            #u_prev_total = np.zeros((10, 2))
            u_prev_total = np.zeros((trajectory_length, self.appx_mpc.mpc.model.n_u))
            mpc.set_initial_guess()

            #mpc.set_tvp_fun(tvp_fun)
            #mpc.set_tvp_fun(self.simulator.tvp_fun)

            start = timer()
            mpc.reset_history()
            self.simulator.reset_history()
            self.estimator.reset_history()

            # set initial values and guess

            mpc.x0 = x0
            self.simulator.x0 = x0
            self.estimator.x0 = x0

            mpc.set_initial_guess()
            u_prev_curr = u_prev
            # run the closed loop for 150 steps
            for k in range(trajectory_length):
                u_prev_total[k] = u_prev_curr.reshape((2,))
                u0 = mpc.make_step(x0)
                u_prev_curr = u0
                if mpc.solver_stats["success"] == False:
                    break
                y_next = self.simulator.make_step(u0)
                x0 = self.estimator.make_step(y_next)

            # we return the complete data structure that we have obtained during the closed-loop run

            end = timer()

            stats = {}
            stats["t_make_step"] = end - start
            stats["success"] = mpc.solver_stats["success"]
            stats["iter_count"] = mpc.solver_stats["iter_count"]

            if "t_wall_total" in mpc.solver_stats:
                stats["t_wall_total"] = mpc.solver_stats["t_wall_total"]
            else:
                stats["t_wall_total"] = np.nan

                # if return_full:
                #    ### get solution
                #    nlp_sol, p_num = nlp_handler.get_mpc_sol(mpc)
                #    z_num = nlp_handler.extract_numeric_primal_dual_sol(nlp_sol)
                ### reduced solution
                # z_num, p_num = nlp_handler.get_reduced_primal_dual_sol(nlp_sol,p_num)
                #    return u0, stats, np.array(z_num), np.array(p_num), mpc.data
                # else:
            return self.simulator.data, stats, u_prev_total

        #def sample_function(x0, u_prev, p):
        #    return run_mpc_closed_loop(x0, u_prev, p)
        def sample_function(x0, u_prev):
            return run_mpc_closed_loop(x0, u_prev)

        # %% Sampling Plan
        # Import sampling plan
        # with open(data_dir+sampling_plan_name+'.pkl','rb') as f:
        with open(data_dir.joinpath(sampling_plan_name + '.pkl'), 'rb') as f:
            plan = pkl.load(f)

        # %% Sampler
        sampler = do_mpc.sampling.Sampler(plan)
        sampler.data_dir = str(samples_dir) + '/'
        sampler.set_param(overwrite=overwrite_sampler)
        sampler.set_param(sample_name=sample_name)

        sampler.set_sample_function(sample_function)

        # %% Main - Sample Data
        # if test_run:
        #    sampler.sample_idx(0)
        # else:
        sampler.sample_data()

        # %% Data Handling
        dh = do_mpc.sampling.DataHandler(plan)

        dh.data_dir = str(samples_dir) + '/'
        dh.set_param(sample_name=sample_name)
        dh.set_post_processing('u0', lambda x: x[0]['_u'])
        dh.set_post_processing('x0', lambda x: x[0]['_x'])
        dh.set_post_processing('u_prev', lambda x: x[2])
        dh.set_post_processing('status', lambda x: x[1]["success"])
        dh.set_post_processing('t_make_step', lambda x: x[1]["t_make_step"])
        dh.set_post_processing('t_wall', lambda x: x[1]["t_wall_total"])
        dh.set_post_processing('iter_count', lambda x: x[1]["iter_count"])
        df = pd.DataFrame(dh[:])
        n_data = df.shape[0]
        df.to_pickle(str(data_dir) + '/' + data_file_name + '_n{}'.format(n_data) + '_all' + '.pkl')
        # %% Save
        # Filter opt and Save
        df = pd.DataFrame(dh.filter(output_filter=lambda status: status == True))
        n_data_opt = df.shape[0]
        df.to_pickle(str(data_dir) + '/' + data_file_name + '_n{}'.format(n_data) + '_opt' + '.pkl')

        # end of function
        return None