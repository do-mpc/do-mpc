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
# Start CasADi
from casadi import *
# Import numpy (a matlib-like scientific toolbox)
import numpy as NP
def dompc():
    
    states = NP.resize(NP.array([]),(end_time/t_step,nx))
    control= NP.resize(NP.array([]),(end_time/t_step,nu))
    alg	   = NP.resize(NP.array([]),(end_time/t_step,1))
    time   = NP.resize(NP.array([]),(end_time/t_step))
    cost   = NP.resize(NP.array([]),(end_time/t_step))
    ref   = NP.resize(NP.array([]),(end_time/t_step))
    cpu    = NP.resize(NP.array([]),(end_time/t_step))
    bias    = NP.resize(NP.array([]),(end_time/t_step))
    parameters = NP.resize(NP.array([]),(end_time/t_step,np))
    states[0,:] = x0
    control[0,:]= u0
    time[0]		= 0
    return states, control, alg, time, cost, ref, cpu, bias, parameters, 