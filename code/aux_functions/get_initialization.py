# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 08:26:40 2013

@author: Sergio Lucia (TU Dortmund)
This files generates a good initial condition for all the collocation points for the first iteration
"""

vars_init_ = NP.zeros(NV)
t0_initialization = 0
tf_initialization = t_step/ni/(deg+1)
x0_initialization = NP.resize(NP.array([]),(nk+1,nx))
x0_initialization = x0
u_initialization = NP.squeeze([u_init0])
vars_init_[X_offset[0,0]  :  X_offset[0,0] + nx] = x0
        
# Skip the initialization for  the initial condition
offset = nx     
# Loop over prediction horizon (stage), nodes in the current stage, 
# children nodes for each node, finite elements and collocation points
for k in range(nk):

   for s in range(n_scenarios[k]):
       for b in range(n_branches[k]):
          #first_j = 1 
          for i in range(ni):
              for j in range(deg+1):
                    # Set integrator
                    integrator.setOption("t0",t0_initialization)
                    integrator.setOption("tf",tf_initialization)            
                    integrator.setInput(vertcat([u_initialization,p_scenario[b+branch_offset[k][s]]]),INTEGRATOR_P)
                    integrator.setInput(x0_initialization,INTEGRATOR_X0)   
                    integrator.evaluate()
                    x_next = NP.squeeze(integrator.output())
                    #pdb.set_trace()
                    vars_init_[X_offset[k,s] + offset   :  X_offset[k,s] + nx + offset]   = x_next
                    offset += nx
                    t0_initialization = tf_initialization
                    tf_initialization = tf_initialization + t_step/ni/(deg+1)
                    x0_initialization = x_next   
  
          vars_init_[X_offset[k,s] + offset : X_offset[k,s] + nx + offset] = x_next
          offset +=nx
       offset = 0      
for  s in range(n_scenarios[nk]):
    vars_init_[X_offset[nk,s] + offset   :  X_offset[nk,s] + nx + offset]   = x_next  
    
# Update the initial guess for the solver
solver.setInput(vars_init_,NLP_SOLVER_X0)
