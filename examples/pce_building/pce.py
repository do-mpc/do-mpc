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
import mcint
import random
import math
import pdb
import time
import matplotlib.pyplot as plt
print ("Starting main script ...")
class stochastic_variable:
    """A class for the definition of stocahstic variables. Define the variable with the following arguments: theta_class = stochastic_variable(theta_casadi, deg, dist, *param)"""
    def __init__(self, theta_casadi, deg, dist, *param):
        # TODO Check weights of beta
         self.symbol = theta_casadi
         # deg denotes the desired degree of the expansion
         self.deg = deg
         self.dist = dist
        # Parse for beta distrubtions
         if dist == "beta":
          if len(param) == 2:
              self.alpha = param[0]
              self.beta = param[1]
          else:
              raise Exception("Specify alfa and beta parameters")
         # Add the domain of each distribution
         # and
         # Add the weighting function of the corresponding pdf
         if dist == "gaussian":
             #self.domain = [-NP.inf, NP.inf]
             # NOTE The domain is truncated to facilitate integration
             self.domain = [-6, 6]
             self.weight_pdf = 1. / (sqrt(2*pi)) * exp(-theta_casadi**2/2)
         elif dist == "uniform":
             self.domain = [-1.,1.]
             self.weight_pdf = 1.0 / 2.
         elif dist == "beta":
             self.domain = [-1.,1.]
             self.weight_pdf = (theta_casadi + 1)**self.beta * (1 - theta_casadi)**self.alpha / ((2**(self.alpha + self.beta + 1)) * BETA_function(self.alpha + 1, self.beta + 1))
         else:
            raise Exception("Unknown distribution: choose gaussian, uniform or beta")


def BETA_function(p , q):
    result = gamma(p) * gamma(q) /(gamma(p+q))
    return result

def legendre_rodriguez(deg,theta):
    # Legendre polynomials for theta \in [-1,1]
    phi_first_term = 1.0 /(2.0 ** deg  * NP.math.factorial(deg))
    deriv_fcn = Function("deriv_fcn",[theta],[(theta**2-1)**deg])
    aux_deriv_fcn = deriv_fcn
    for i in range(deg):
        aux_deriv_fcn = aux_deriv_fcn.jacobian()
    phi = phi_first_term * aux_deriv_fcn.call([theta])[0]
    return phi

def hermite_rodriguez(deg,theta):
    # Hermite polynomials with **unit** variance
    # Fisher Diss: An additional method for using a non-unit variance is to
    # multiply  by  everywhere it appears in the actual equations
    phi_first_term = (-1.0)**deg * exp(theta ** 2. / 2.)
    deriv_fcn = Function("deriv_fcn",[theta],[exp(-theta**2/2.)])
    aux_deriv_fcn = deriv_fcn
    for i in range(deg):
        aux_deriv_fcn = aux_deriv_fcn.jacobian()
    phi = phi_first_term * aux_deriv_fcn.call([theta])[0]
    return phi

def jacobi_rodriguez(deg,theta, alpha, beta):
    # Jacobi polynomials for theta \in [-1,1]
    phi_first_term = (-1.0)**deg / (2.0 ** deg  * NP.math.factorial(deg) *(1-theta)**alpha *(1+theta)**beta)
    deriv_fcn = Function("deriv_fcn",[theta],[(1-theta)**(deg+alpha) * (1+theta)**(deg+beta)])
    aux_deriv_fcn = deriv_fcn
    for i in range(deg):
        aux_deriv_fcn = aux_deriv_fcn.jacobian()
    phi = phi_first_term * aux_deriv_fcn.call([theta])[0]
    return phi

def phi_mono(deg, theta_class):
    # deg gives the corresponding polynomial
    # theta_class.deg contains the order of the approximation (not equal to the argument of this function)
    dist = theta_class.dist
    theta = theta_class.symbol
    if dist == "gaussian":
        phi = hermite_rodriguez(deg,theta)
    elif dist == "uniform":
        phi = legendre_rodriguez(deg, theta)
    elif dist == "beta":
        phi = jacobi_rodriguez(deg,theta,theta_class.alpha,theta_class.beta)
    else:
        raise Exception("Unknown distribution: choose gaussian, uniform or beta")
    return phi


def poly_basis(theta_class_vector):
    # CHECK this funciton is correct
    # Make the products of all the monomials for each element of theta so that the maximum degree is deg
    # theta_stoc_vector should be a vector foe elements from the class stochastic_variable
    # poly_basis(['x1', 'x2'], ['y1', 'y2']) --> x1, x2, y1, y2, x1y1
    # Al polynomials shouπdl be of the same dimension
    # TODO Parsing to check same dimension of polynomials
    # Get the number of uncertainties and degree of approximation
    m = len(theta_class_vector)
    deg = []
    dist = []
    theta = []
    poly_vector = []
    for j in range(m):
        deg.append(theta_class_vector[j].deg)
        dist.append(theta_class_vector[j].dist)
        theta.append(theta_class_vector[j])
        # Initialize list to contain all degree polynomials for each uncertainty
        poly_vector.append([])
        args_poly = []
    # TODO Total number of elements of the expansion for different degrees for each uncertainty
    #L = NP.math.factorial(m + deg) / (NP.math.factorial(m) * NP.math.factorial(deg))
    for j in range(m):
        for i in range(1,deg[j]+1):
            poly_vector[j].append([phi_mono(i,theta[j])])
        args_poly.append(poly_vector[j])
    # The list poly_vector contains all the monovariate polynomials for each uncertainty
    # Now all the combinations up to degree theta_class_vector.deg should be performed
    # args will contain indices to check for the total sum of the degree of mixed terms
    args = []
    for i in range(m):
        args.append(range(1,deg[0]+1))
    # Here start the combinations of all the elements of each polynomial
    pools = map(tuple, args)
    result_yield = []
    result_poly_yield = []
    # The list result contains indices (degrees) and result_poly the actual polynomials
    result = [[]]
    result_poly = [[]]
    for idx_p, pool in enumerate(pools):
        for idx_x, x in enumerate(result):
            for idx_y, y in enumerate(pool):
                if x == []:
                    # For the first iteration always add
                    candidate = 0
                else:
                    # sum all the previous degrees
                    prev_degree = 0
                    for i in range(len(x)):
                        prev_degree = prev_degree + int(x[i])
                    candidate = prev_degree + int(y)
                # Check that the overall order of the crossed terms is less than the desired one
                if candidate > deg[0]:
                    pass
                else:
                    result = result + [x+[str(y)]]
                    result_poly = result_poly + [result_poly[idx_x] + [args_poly[idx_p][idx_y]]]
    for prod in result_poly:
        result_poly_yield.append(tuple(prod))
    # Now build the multiplications of the polynomials that are included in the list result_poly_yield
    pce_basis = []
    for i in range(1, len(result_poly_yield)):
        # The first term is common to all of them
        aux = result_poly_yield[i][0][0]
        # Multiply all the elements in the same position of the list
        for j in range(len(result_poly_yield[i]) - 1):
            aux = mtimes(aux,result_poly_yield[i][j+1][0])
        pce_basis.append(aux)
    # Add the first element of the expansion (1)
    pce_basis.insert(0,1)
    return pce_basis

# Define the functions for perform the multidimensional integrals
def sampler(theta_class_vector):
    while True:
        res = []
        for i in range(len(theta_class_vector)):
            if theta_class_vector[i].dist == "gaussian":
                random_gen = random.gauss(0,1)
            if theta_class_vector[i].dist == "uniform":
                random_gen = random.uniform(-1,1)
            if theta_class_vector[i].dist == "beta":
                alpha = theta_class_vector[i].alpha
                beta = theta_class_vector[i].beta
                random_gen = random.betavariate(alpha,beta)
            res.append(random_gen)
        yield (res)

def integrand_nquad(*x):
    # pdb.set_trace()
    # This function defines the integrand for a multidimensional integration
    # There are always five more elements as variables to integrate
    # (one for the vector of objects and three polynomial terms and the weighting function)
    n_elem = len(x) - 5
    # The first n_elem contain the variables to integrate, the element n_elem the stochastic variables objects
    theta_class_vector = x[n_elem]
    # Extract the rest of terms
    # poly_1 = x[n_elem + 1]
    # poly_2 = x[n_elem + 2]
    # poly_3 = x[n_elem + 3]
    # weight_pce = x[n_elem + 4]

    # Calculate the integrand
    #res = poly_1 * poly_2 * poly_3 * weight_pce
    res = x[n_elem + 1] * x[n_elem + 2] * x[n_elem + 3] * x[n_elem + 4]
    # Substitute into the CasADi object
    # TODO: Having this inside the integrand is probably highly inefficient
    for i in range(n_elem):
        res = substitute(res,theta_class_vector[i].symbol,x[i])
    return res


#integrate.nquad(integrand_numerator, limits, args = (theta_class_vector,1,1,2))
#integrate.nquad(integrand_nquad, limits, args = params)


def Psi(idx, theta_class_vector, pce_basis, weight_pce, limits, ab_denominator):
    # define functions for the integration with scipy
    p = len(pce_basis)  # This is the order of the expansion
    Psi = NP.zeros([p,p])
    for i in range(p):
        for j in range(i,p): # Build here only the upper triangular part
            # Choose the external parameters to provide to the integration
            params_num = (theta_class_vector, pce_basis[i], pce_basis[idx], pce_basis[j], weight_pce)
            # integration options
            options = {'epsrel': 1.0e-6}
            e_hat_numerator, error_quad = integrate.nquad(integrand_nquad, limits, args = params_num)
            # If the numerator is 0 do not integrate to calculate the denominator
            if abs(e_hat_numerator) < 1e-8:
                e_hat_numerator = 0
                #e_hat_denominator = 1.
                #Psi[i,j] = 0
            else:
                pass
                #params_denom = (theta_class_vector, pce_basis[i], pce_basis[i], pce_basis[0], weight_pce)
                #e_hat_denominator, error_quad	= integrate.nquad(integrand_nquad, limits, args = params_denom)
            #Psi[i,j] = e_hat_numerator/ab_denominator[i]
            Psi[i,j] = e_hat_numerator / ab_denominator[i]
            if j != i:
                Psi[j,i] = Psi[i,j]  # fill the lower triangular part
    return Psi


# Define each one of the submatrices Aij and Bij (Fisher Dissertation, III.26,28)
def A_ij_calc(i , j,  theta_class_vector, pce_basis, weight_pce, limits, A_model, ab_denominator, Psi_symbol):
    p = len(pce_basis)
    A_ij = NP.zeros([p,p])
    for k in range(p):
        params_num = (theta_class_vector, A_model[i,j], pce_basis[k], 1.0, weight_pce)
        # Do not integrate if the element of the matrix is 0
        if SX(A_model[i,j]).is_zero():
            a_ij_numerator = 0
        else:
            a_ij_numerator, error_quad = integrate.nquad(integrand_nquad, limits, args = params_num)
        if abs(a_ij_numerator) < 1e-7:
            a_ij_numerator = 0
            a_ij = 0
        else:
            a_ij = a_ij_numerator / ab_denominator[k]
        A_ij += a_ij * Psi_symbol[k]

    return A_ij

def B_ij_calc(i , j,  theta_class_vector, pce_basis, weight_pce, limits, B_model, ab_denominator, Psi_symbol):
    p = len(pce_basis)
    B_ij = NP.zeros([p,p])
    # Calculation of B_ij
    for k in range(p):
        params_num = (theta_class_vector, B_model[i,j], pce_basis[k], 1.0, weight_pce)
        if SX(B_model[i,j]).is_zero():
            b_ij_numerator = 0
        else:
            b_ij_numerator, error_quad = integrate.nquad(integrand_nquad, limits, args = params_num)
        # Do not compute other terms if the numerator is 0
        if abs(b_ij_numerator) < 1e-7:
            b_ij_numerator = 0
            b_ij = 0
        else:
            b_ij = b_ij_numerator / ab_denominator[k]
        B_ij += b_ij * Psi_symbol[k]
        # Alternative calulation according to Paulson CDC 2014-2015
        #B_ij += b_ij * Psi_symbol[0]
    return B_ij
