'''
Neural network approxmation of the model predictive controller.
'''

import warnings
from .. import __TORCH_INSTALLED__

if __TORCH_INSTALLED__:
    from ._trainer import Trainer
    from ._ampc import ApproxMPC, FeedforwardNN, ApproximateMPCSettings
    from ._ampc_sampler import AMPCSampler
else:
    warnings.warn('The approximateMPC feature requires PyTorch, which is not installed by default. Please install the full version of do-mpc to use this feature: pip install do-mpc[full]')
