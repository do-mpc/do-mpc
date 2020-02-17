# do mpc
<img src="https://github.com/do-mpc/DO-MPC/blob/master/documentation/logo_v2.png" width="200" alt="do-mpc_logo." 
  align="right">
**do mpc** proposes a new, modularized implementation and testing support for optimal control schemes based on  MPC approaches. The goal of this software project is to offer a simple to use and efficient platform, which allows users to define and test their  problems very fast and trouble-free. In most cases, such implementations are highly complex and cumbersome, requiring considerable coding effort that only produces hardcoded solutions for each individual test case. With **do mpc** we propose a generalized approach based on simple templates  that can be edited for each individual problem. A robust and time efficient core module combines everything together automatically, such that the coding effort is reduced drastically. Taking advantage of state of the art third party software, **do-mpc** is able to handle a wide variety of problems, making even large systems real time feasible.

Moreover, **do mpc** provides a very simple framework for the implementation of a state-of-the art robust nonlinear model predictive control approach called multi-stage NMPC, which is based on the description of the uncertainty as a scenario tree.

The **do mpc** software is Python based and works therefore on any OS with a Python 3.x distribution. **do-mpc** has been developed at the DYN chair of the TU Dortmund by Sergio Lucia and Alexandru Tatulea.

# Installation instructions
For detailed instructions go to [readthedocs](https://do-mpc.readthedocs.io)

# FAQ
Go [here](https://github.com/do-mpc/do-mpc/wiki/FAQ) for a list of the most frequently asked questions

# Citing do-mpc
If you use do-mpc for published work please cite it as:

S. Lucia, A. Tatulea-Codrean, C. Schoppmeyer, and S. Engell. Rapid development of modular and sustainable nonlinear model predictive control solutions. Control Engineering Practice, 60:51-62, 2017

Please remember to properly cite other software that you might be using too if you use do-mpc (e.g. CasADi, IPOPT, ...)
