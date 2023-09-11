'''
A OPC UA wrapper for do-mpc.
'''

import warnings
from .. import __ASYNCUA_INSTALLED__

if __ASYNCUA_INSTALLED__:
    from ._server import RTServer
    from ._client import RTClient
    from ._base import RTBase
    from ._helper import NamespaceEntry, Namespace, ServerOpts, ClientOpts
else:
    warnings.warn('The opcua feature is not available. Please install the full version of do-mpc to access this feature.')
