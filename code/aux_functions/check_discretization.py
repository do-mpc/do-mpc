# This file plots the difference between the collocation points at the nodes and the integration using sundials with a very tight accuracy for the nominal case

# Define the integrator for the nominal model and only the first time
t0_check = t0_sim
tf_check = tf_sim
if index_mpc == 1:
  integrator_check = CVodesIntegrator(f_sim)
  integrator_check.setOption("abstol",1e-10) # tolerance
  integrator_check.setOption("reltol",1e-10) # tolerance
  integrator_check.setOption("steps_per_checkpoint",100)
  integrator_check.setOption("t0",t0_check)
  integrator_check.setOption("tf",tf_check)
  integrator_check.setOption("fsens_abstol",1e-8)
  integrator_check.setOption("fsens_reltol",1e-8)
  #integrator_check.setOption("asens_abstol",1e-8)
  #integrator_check.setOption("asens_reltol",1e-8)
  integrator_check.setOption("exact_jacobian",True)
  integrator_check.init()
 
# Define data structures
x0_check    = NP.resize(NP.array([]),(nk+1,nx))
x0_check[0,:] = x0_sim
error_check = NP.resize(NP.array([]),(nk+1,nx))
time_check  = NP.resize(NP.array([]),(nk+1))

# Integrate for all the prediction horizon
for counter_check in range(nk):
  # Set the inputs fot the integrator
  u_check = NP.squeeze(v_opt[U_offset[counter_check,0]:U_offset[counter_check,0]+nu])
  integrator_check.setInput(u_check,INTEGRATOR_P)
  integrator_check.setInput(x0_check[counter_check,:],INTEGRATOR_X0)
  # Integrate one sampling time
  integrator_check.evaluate()
  # Get the ouput and store it
  x0_check[counter_check + 1,:]    = NP.squeeze(integrator_check.output())
  error_check[counter_check + 1,:] = NP.squeeze(v_opt[X_offset[counter_check,0]:X_offset[counter_check,0]+nx])-NP.squeeze(x0_check[counter_check,:])
  time_check[counter_check + 1] = tf_check
  t0_check = tf_check
  tf_check = tf_check + t_step
  integrator_check.setOption("t0",t0_check)
  integrator_check.setOption("tf",tf_check)

# Plot the errors

plt.ion()
# Plot MPC results
fig = plt.figure(5,figsize=(8,8))
#plt.clf()
plt.subplot(421)
#plt.clf()
plt.plot(time_check, error_check[:,0]*x_scaling[0])
plt.ylabel("$error x0",fontweight="bold")
#plt.xlabel('Time [h]')
#plt.axis([0,tf_sim,363.15-temp_range-273.15-1,363.15+temp_range+1-273.15])
#plt.legend(['Tracking NMPC'])
plt.grid()

plt.subplot(422)
#plt.clf()
plt.plot(time_check, error_check[:,1]*x_scaling[1])
plt.ylabel("$error x0",fontweight="bold")
#plt.xlabel('Time [h]')
#plt.axis([0,tf_sim,363.15-temp_range-273.15-1,363.15+temp_range+1-273.15])
#plt.legend(['Tracking NMPC'])
plt.grid()

plt.subplot(423)
#plt.clf()
plt.plot(time_check, error_check[:,2]*x_scaling[2])
plt.ylabel("$error x0",fontweight="bold")
#plt.xlabel('Time [h]')
#plt.axis([0,tf_sim,363.15-temp_range-273.15-1,363.15+temp_range+1-273.15])
#plt.legend(['Tracking NMPC'])
plt.grid()

plt.subplot(424)
#plt.clf()
plt.plot(time_check, error_check[:,3]*x_scaling[3])
plt.ylabel("$error x0",fontweight="bold")
#plt.xlabel('Time [h]')
#plt.axis([0,tf_sim,363.15-temp_range-273.15-1,363.15+temp_range+1-273.15])
#plt.legend(['Tracking NMPC'])
plt.grid()

plt.subplot(425)
#plt.clf()
plt.plot(time_check, error_check[:,4]*x_scaling[4])
plt.ylabel("$error x0",fontweight="bold")
#plt.xlabel('Time [h]')
#plt.axis([0,tf_sim,363.15-temp_range-273.15-1,363.15+temp_range+1-273.15])
#plt.legend(['Tracking NMPC'])
plt.grid()

plt.subplot(426)
#plt.clf()
plt.plot(time_check, error_check[:,5]*x_scaling[5])
plt.ylabel("$error x0",fontweight="bold")
#plt.xlabel('Time [h]')
#plt.axis([0,tf_sim,363.15-temp_range-273.15-1,363.15+temp_range+1-273.15])
#plt.legend(['Tracking NMPC'])
plt.grid()

plt.subplot(427)
#plt.clf()
plt.plot(time_check, error_check[:,6]*x_scaling[6])
plt.ylabel("$error x0",fontweight="bold")
#plt.xlabel('Time [h]')
#plt.axis([0,tf_sim,363.15-temp_range-273.15-1,363.15+temp_range+1-273.15])
#plt.legend(['Tracking NMPC'])
plt.grid()

plt.subplot(428)
#plt.clf()
plt.plot(time_check, error_check[:,7]*x_scaling[7])
plt.ylabel("$error x0",fontweight="bold")
#plt.xlabel('Time [h]')
#plt.axis([0,tf_sim,363.15-temp_range-273.15-1,363.15+temp_range+1-273.15])
#plt.legend(['Tracking NMPC'])
plt.grid()
plt.draw()
