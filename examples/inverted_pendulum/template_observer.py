# 	 -*- coding: utf-8 -*-
#
#    This file is part of DO-MPC
#
#    DO-MPC: An environment for the easy, modular and efficient implementation of
#            robust nonlinear model predictive control
#
#	 The MIT License (MIT)
#
#    Copyright (c) 2014-2018 Sergio Lucia, Alexandru Tatulea-Codrean
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

from casadi import *
import core_do_mpc
def observer(model):

	# Full state feedback
	observer_dict = {'x':1}
	observer = core_do_mpc.observer(model,observer_dict)
	# here some functions depending on observer_1

	# Implement here your own observer

	"""
	--------------------------------------------------------------------------
	template_observer: pass information (not necessary to edit)
	--------------------------------------------------------------------------
	"""
	return observer
