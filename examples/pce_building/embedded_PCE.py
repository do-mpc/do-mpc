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


# This script generates the PCE expansion of a linear system using the Galerkin
# projection method

# Define the polynomial basis for a given degree, a number of uncertainties theta.size()
# and a given distribution
print ("Loading CasADi ...")
from casadi import *
print ("Loading other packages ...")
import numpy as NP
import itertools
from scipy.special import gamma
from scipy import integrate
import scipy.io
import mcint
import random
import math
import pdb
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pce

def expanded_building():
    # Definition of the stochastic variables
    theta_1 = SX.sym("theta_1")
    theta_2 = SX.sym("theta_2")

    theta_1_class = pce.stochastic_variable(theta_1,0,"uniform")
    # theta_1_class = stochastic_variable(theta_1,2,"beta",2,2)
    theta_2_class = pce.stochastic_variable(theta_2,0,"uniform")
    #theta_class_vector = [theta_1_class,theta_2_class]
    theta_class_vector = [theta_1_class]


    x1 = SX.sym("x1")
    x2 = SX.sym("x2")
    n_steps = 30
    sample_MC = 5000
    x_initial = NP.array([1.0,1.0])*3
    x_original_initial = x_initial
    u_chosen = NP.array([1.0, 10, 10.0, 5.0])

    # A_model = SX(NP.array([[0.9*(1+theta_1**4/3.33333),-0.4],[0.1, 0.5]]))# unstable
    # B_model = SX(NP.array([0.2,0.2]))

    # Building Energy Management system
    A_model = SX(NP.array([[1.0,0],[0, 1.0]]))
    B_model = SX(NP.array([[0.0035*(1+theta_1), 0, 0],[0, -5.0*(1+theta_1), 0]]))
    E_model = SX(1e-3*NP.array([[22.217, 1.7912, 42.212],[0,0,0]]))

    nx = A_model.size1()
    nu = B_model.size2()
    nd = E_model.size2()
    x = SX.sym("x",nx)
    u = SX.sym("u",nu)
    d = SX.sym("d",nd)
    export_name = 'pce_analytic'
    u_pred = SX.sym("u_pred",n_steps)
    #rhs = Function('rhs',[x1, u, theta_1],
    #[1.0*(theta_1*0.5 + 0.5)* x1 + u])
    rhs = mtimes(A_model,x) + mtimes(B_model,u) #+ mtimes(E_model, d)
    rhs_fcn = Function('rhs',[x, u, theta_1],[rhs])

    weight_pce = 1.0
    for i in range(len(theta_class_vector)):
        weight_pce = weight_pce * theta_class_vector[i].weight_pdf

    # Choose the limits of integration
    limits = []
    for i in range(len(theta_class_vector)):
        limits.append(theta_class_vector[i].domain)

    pce_basis = pce.poly_basis(theta_class_vector)
    pce_terms = len(pce_basis)

    # Build A_extended and B_extended
    A_extended = NP.zeros([nx*(pce_terms),nx*(pce_terms)])
    B_extended = NP.zeros([nx*(pce_terms),nu*(pce_terms)])
    E_extended = NP.zeros([nx*(pce_terms),nd*(pce_terms)])

    ab_denominator = []
    Psi_symbol = []
    for k in range(pce_terms):
        print "Calculating denominator of order %d " % (k)
        # Calculation of the denominators, which are common to all terms
        params_denom = (theta_class_vector, pce_basis[k], pce_basis[k], 1.0, weight_pce)
        ab_denominator_, error_quad = integrate.nquad(pce.integrand_nquad, limits, args = params_denom)
        ab_denominator.append(ab_denominator_)
    for k in range(pce_terms):
        print "Calculating Psi of order %d " % (k)
        # Calculate all pce_basis which are then used on the expansion of the matrices
        Psi_symbol_ = pce.Psi(k, theta_class_vector, pce_basis, weight_pce, limits, ab_denominator)
        Psi_symbol.append(Psi_symbol_)

    for i in range(nx):
        for j in range(nx):
            print "Calculating element (%d,%d) of A_extended" % (i,j)
            A_extended[i*(pce_terms):(i+1)*(pce_terms),j*(pce_terms):(j+1)*(pce_terms)] = pce.A_ij_calc(i , j,  theta_class_vector, pce_basis, weight_pce, limits, A_model, ab_denominator, Psi_symbol)
    for i in range(nx):
        for j in range(nu):
            print "Calculating element (%d,%d) of B_extended" % (i,j)
            B_extended[i*(pce_terms):(i+1)*(pce_terms),j*(pce_terms):(j+1)*(pce_terms) ] = pce.B_ij_calc(i , j,  theta_class_vector, pce_basis, weight_pce, limits, B_model, ab_denominator, Psi_symbol)
    for i in range(nx):
        for j in range(nd):
            print "Calculating element (%d,%d) of E_extended" % (i,j)
            E_extended[i*(pce_terms):(i+1)*(pce_terms),j*(pce_terms):(j+1)*(pce_terms) ] = pce.B_ij_calc(i , j,  theta_class_vector, pce_basis, weight_pce, limits, E_model, ab_denominator, Psi_symbol)


    # For the moment do not consider an expanded u or d
    B_extended = B_extended[:,0::pce_terms] # This means: take only the column for the first coeff
    E_extended = E_extended[:,0::pce_terms]

    NP.savetxt("A_extended.txt", A_extended)
    NP.savetxt("B_extended.txt", B_extended)
    NP.savetxt("E_extended.txt", E_extended)

    # x_extended = SX.sym("x_extended",nx*pce_terms)
    # x0_extended = NP.zeros([nx*(pce_terms)])
    # x0_extended[0] = x_original_initial[0]
    # x0_extended[pce_terms] = x_original_initial[1]
    # x_initial = x0_extended
    # # Do not consider for the moment an expanded u
    # # pdb.set_trace()
    # u_extended = []
    # for index_u in range(nu):
    #     for index_u2 in range(pce_terms):
    #         if index_u2 == 0:
    #             u_extended.append(u[index_u])
    #         else:
    #             u_extended.append(0)
    # # B_extended =  NP.array([B_extended[:,0:nu]]).T
    # u_extended = NP.array(u_extended)
    # pce_system = mtimes(A_extended,x_extended) + mtimes(B_extended,u_extended)
    # pce_system_fcn = Function('pce_system',[x_extended, u],[pce_system])
    # pce_coefficients = NP.resize(NP.array([]),(nx*pce_terms,n_steps+1))
    # pce_coefficients[:,0] = x0_extended
    # for i in range(1,n_steps+1):
    #     x_next = pce_system_fcn(x_initial,u_chosen)
    #     x_initial = x_next
    #     pce_coefficients[:,i] = NP.squeeze(x_next)
    # # Compute the moments based on pce coefficients
    # pce_mean = NP.resize(NP.array([]),(nx,n_steps+1))
    # pce_variance = NP.resize(NP.array([]),(nx,n_steps+1))
    # for j in range(0,n_steps+1):
    #     for i in range(nx):
    #         pce_mean[i,j] = pce_coefficients[i*pce_terms][j]
    #         aux_sum = 0
    #         for k in range(1,pce_terms):
    #             aux_sum = aux_sum + pce_coefficients[i*pce_terms + k][j]**2 * ab_denominator[k]
    #         pce_variance[i,j] = aux_sum
    #
    #
    #
    # # Compute analytic moments
    #
    # integrand_nquad = pce.integrand_nquad
    # analytic_mean = NP.resize(NP.array([]),(nx,n_steps+1))
    # analytic_variance = NP.resize(NP.array([]),(nx,n_steps+1))
    # analytic_mean[:,0] = x_original_initial
    # analytic_variance[:,0] = NP.zeros(nx)
    # for i in range(1,n_steps+1):
    #     x_next = rhs_fcn(x_original_initial,u_chosen,theta_1)
    #     x_initial = x_next
    #     for j in range(nx):
    #         params_exp = (theta_class_vector, x_next[j], 1.0, 1.0, weight_pce)
    #         expectation, error_quad = integrate.nquad(integrand_nquad, limits, args = params_exp)
    #         analytic_mean[j,i] = expectation
    #         params_var = (theta_class_vector, x_next[j]**2, 1.0, 1.0, weight_pce)
    #         var, error_quad = integrate.nquad(integrand_nquad, limits, args = params_var)
    #         analytic_variance[j,i] = var - analytic_mean[j,i]**2
    #
    #
    # # Calculation with Monte Carlo sampling
    # x_MC = NP.resize(NP.array([]),(sample_MC, n_steps + 1, len(x_original_initial)))
    # random.seed()
    #
    # for index_MC in range(sample_MC):
    #     theta_real = []
    #     x_initial_real = x_original_initial
    #     A_real = A_model
    #     B_real = B_model
    #     # Sample the random variables, which stay constant
    #
    #     for i in range(len(theta_class_vector)):
    #         if theta_class_vector[i].dist == "gaussian":
    #             # By default mu = 0, sigma = 1.0
    #             theta_real.append(random.gauss(0.0,1.0))
    #         elif theta_class_vector[i].dist == "uniform":
    #             theta_real.append(random.uniform(-1.0,1.0))
    #         elif theta_class_vector[i].dist == "beta":
    #             theta_real.append(random.betavariate(theta_class_vector[i].alpha,theta_class_vector[i].beta))
    #         elif theta_class_vector[i].dist == "dirac_splitted":
    #             theta_real.append(random.choice([-1.0,1.0]))
    #         A_real = substitute(A_real, theta_class_vector[i].symbol, theta_real[i])
    #         B_real = substitute(B_real, theta_class_vector[i].symbol, theta_real[i])
    #     # Initial condition
    #     x_MC[index_MC][0][:] = NP.squeeze(x_initial_real)
    #     for index_nk in range(1,n_steps+1):
    #         #u_real = mtimes(K_feedback,x_initial_real)
    #         x_next_real = mtimes(A_real,x_initial_real) + mtimes(B_real,u_chosen)
    #         #x_next_real = rhs(x_initial_real[0],u_real[index_nk - 1],theta_real[0])
    #         x_initial_real = x_next_real
    #         for jj in range(len(x_original_initial)):
    #             x_MC[index_MC][index_nk][jj] = x_next_real[jj]
    # mean_MC = NP.mean(x_MC, axis = 0)
    # var_MC = NP.var(x_MC, axis = 0)
    #
    # plt.ion()
    # fig = plt.figure(1)
    # plot = plt.subplot(2, 1, 1)
    # # plt.plot(analytic_mean[0,:], linewidth=2, color = '0')
    # plt.plot(pce_mean[0,:], linewidth=2, color = '0.6')
    # plt.plot(mean_MC[:,0], linewidth=2, color = '0.8')
    # #plt.plot(mean_MC[:,0])
    # plt.ylabel("Mean")
    # #plt.xlabel("Time")
    # plt.grid()
    # plot.yaxis.set_major_locator(MaxNLocator(4))
    #
    # plot = plt.subplot(2, 1, 2)
    # # plt.plot(analytic_variance[0,:], label ="Analytic", linewidth=2, color = '0')
    # plt.plot(pce_variance[0,:], label ="PCE", linewidth=2, color = '0.6')
    # plt.plot(var_MC[:,0], label ="MC", linewidth=2, color = '0.8')
    # #plt.plot(var_MC[:,0], label = 'MC')
    # plt.ylabel("Variance")
    # plt.xlabel("Time step")
    # plt.grid()
    # plt.legend(loc='best')
    # plot.yaxis.set_major_locator(MaxNLocator(4))
    # plt.show()


    return A_extended, B_extended, E_extended, pce_terms
