"""
Tools for machine learning and system identification.

.. warning::
    The :py:class:`ONNXConversion` class is  experimental.
    
"""

import warnings
from .. import __ONNX_INSTALLED__

if __ONNX_INSTALLED__:
    from ._onnxconversion import ONNXConversion, ONNXOperations
else:
    warnings.warn('The ONNX feature is not available. Please install the full version of do-mpc to access this feature.')