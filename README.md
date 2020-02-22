![do_mpc](documentation/source/static/dompc_var_02_rtd_blue.svg)

![Documentation Status](https://readthedocs.org/projects/do-mpc/badge/?version=latest)
[![Build Status](https://travis-ci.org/do-mpc/do-mpc.svg?branch=master)](https://travis-ci.org/do-mpc/do-mpc)

## Introduction

**do mpc** proposes a new, modularized implementation for optimization based model predictive control (MPC) and moving horizon estimation (MHE).
The goal of this software project is to offer a simple to use and efficient platform,
which allows users to define and test their problems very fast and trouble-free.
In most cases, such implementations are highly complex and cumbersome,
requiring considerable coding effort that only produces hard-coded solutions for each individual test case.
With **do mpc** we propose a generalized approach:
The **do mpc** model class is configured to represent the investigated system and is at the core of the **do mpc** simulator, MHE and MPC.
These modules can be easily configured and work independently or in conjunction.

A core feature of **do mpc** is the simple framework for the implementation of a state-of-the art **robust nonlinear model predictive control** approach called multi-stage NMPC, which is based on the description of the uncertainty as a scenario tree.

The **do mpc** software is Python based and works therefore on any OS with a Python 3.x distribution. **do mpc** has been developed at the DYN chair of the TU Dortmund by Sergio Lucia and Alexandru Tatulea.

## Installation instructions
Installation instructions are given [here](https://do-mpc.readthedocs.io/en/latest/installation.html).

## Documentation
Please visit our extensive [documentation](https://do-mpc.readthedocs.io/en/latest/index.html), kindly hosted on readthedocs.

## Citing **do mpc**
If you use **do mpc** for published work please cite it as:

S. Lucia, A. Tatulea-Codrean, C. Schoppmeyer, and S. Engell. Rapid development of modular and sustainable nonlinear model predictive control solutions. Control Engineering Practice, 60:51-62, 2017

Please remember to properly cite other software that you might be using too if you use **do mpc** (e.g. CasADi, IPOPT, ...)
