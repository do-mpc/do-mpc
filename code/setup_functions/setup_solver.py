# 	 -*- coding: utf-8 -*-
#
#    This file is part of DO-MPC
#
#    DO-MPC: An environment for the easy, modular and efficient implementation of
#            robust nonlinear model predictive control
#
#    The MIT License (MIT)
#
#    Copyright (c) 2014-2015 Sergio Lucia, Alexandru Tatulea-Codrean, Sebastian Engell
#                            TU Dortmund. All rights reserved
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.
#

import setup_nlp
from casadi import *
import numpy as NP
import core_do_mpc

def setup_solver(configuration):


    # Call setup_nlp to generate the NLP
    nlp_dict_out = setup_nlp.setup_nlp(configuration.model, configuration.optimizer)

    # Set options
    opts = {}
    opts["expand"] = True
    opts["ipopt.linear_solver"] = configuration.optimizer.linear_solver
    #TODO: this should be passed as parameters of the optimizer class
    opts["ipopt.max_iter"] = 500
    opts["ipopt.tol"] = 1e-6
    # Setup the solver
    solver = nlpsol("solver", configuration.optimizer.nlp_solver, nlp_dict_out['nlp_fcn'], opts)
    arg = {}

    # Initial condition
    arg["x0"] = nlp_dict_out['vars_init']

    # Bounds on x
    arg["lbx"] = nlp_dict_out['vars_lb']
    arg["ubx"] = nlp_dict_out['vars_ub']

    # Bounds on g
    arg["lbg"] = nlp_dict_out['lbg']
    arg["ubg"] = nlp_dict_out['ubg']
    # NLP parameters
    arg["p"] = configuration.model.ocp.u0
    # TODO: better way than adding new fields here?
    configuration.optimizer.solver = solver
    configuration.optimizer.arg = arg
    configuration.optimizer.nlp_dict_out = nlp_dict_out
    return configuration
