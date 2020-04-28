*************************
Moving horizon estimation
*************************


.. math::

    \min_{\textbf{x}_{0:N},\textbf{u}_{0:N-1},\textbf{w}_{0:N-1}}\
    & \underbrace{(x_0 - \tilde{x}_0)^T P_x (x_0 - \tilde{x}_0)}_{\text{arrival cost states}} +
    \underbrace{(p_0 - \tilde{p}_0)^T P_p (p_0 - \tilde{p}_0)}_{\text{arrival cost params.}} \\
    & +\sum_{k=0}^{n-1} \underbrace{(h(x_k, u_k, p_k) - y_k)^T P_{y,k} (h(x_k, u_k, p_k) - y_k)
    + w_k^T P_w w_k}_{\text{stage cost}} \\
    \text{s.t.:}\quad & x_{k+1}=f(x_{k},u_{k},z_{k},p_{k},p_{tv,k}) + w_{k}
