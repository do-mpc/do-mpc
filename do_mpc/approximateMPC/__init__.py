'''
Neural network approxmation of the model predictive controller.
'''

from ._trainer import Trainer
from ._ampc import ApproxMPC, FeedforwardNN, ApproximateMPCSettings
from ._ampc_sampler import AMPCSampler
