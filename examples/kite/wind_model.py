import numpy as np
from scipy.signal import TransferFunction as TF


class Wind:
    def __init__(self, w_ref, t_step, k_sigma_w = 0.14):
        # given parameters
        self.w_ref  =  w_ref          # [m/s]
        z_ref = 10          # [m]
        L_W = 100           # [m]
        Chi = 15            # [°]
        a = 0.15            # [-] surface friction coefficient


        # static parameters
        tau_F = L_W / self.w_ref
        K_F = np.sqrt(1.49*tau_F/t_step)
        self.sigma_w = k_sigma_w * self.w_ref         # standard deviation of w_N
        self.bar_w_N = -self.sigma_w/2/self.w_ref

        # Create transfer function:
        num = [K_F]
        den = [tau_F,1.0]
        H_F_tf_cont = TF(num,den)
        # Convert to state-space model:
        H_F_ss_cont = H_F_tf_cont.to_ss()
        # Convert to discrete-time:
        H_F_ss_dis = H_F_ss_cont.to_discrete(dt=t_step)

        self.A_dis = float(H_F_ss_dis.A)
        self.B_dis = float(H_F_ss_dis.B)
        self.C_dis = float(H_F_ss_dis.C)

        #
        self.f_init = np.random.normal()/4


    def make_step(self):
        w_N_cur = self.bar_w_N + self.sigma_w * self.C_dis * self.f_init
        w_N = self.w_ref + w_N_cur

        new_u = np.random.normal()/4
        self.f_init = self.A_dis * self.f_init + self.B_dis * new_u

        return w_N
