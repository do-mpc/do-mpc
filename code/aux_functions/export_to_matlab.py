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


# This script saves the results to a .mat file that can be loaded using Matlab


def export_to_matlab(mpc_states, mpc_control, mpc_time, mpc_cpu, index_mpc, x_scaling, u_scaling, export_name):
    # FIXME Inprove this script to save also algebraic states
    mpc_states = mpc_states[:index_mpc, :] * x_scaling
    mpc_control = mpc_control[1:index_mpc, :] * u_scaling
    mpc_time = NP.array([mpc_time[:index_mpc]]).T
    mpc_cpu = NP.array([mpc_cpu[1:index_mpc]]).T
    scipy.io.savemat(export_name, mdict={"mpc_states":mpc_states,"mpc_control":mpc_control, "mpc_time":mpc_time, "mpc_cpu":mpc_cpu})
