'''
A OPC UA wrapper for do-mpc.
'''

from ._server import RTServer
from ._client import RTClient
from ._base import RTBase
from ._helper import NamespaceEntry, Namespace, ServerOpts, ClientOpts
try:
    import asyncua.sync as opcua
except ImportError:
    raise ImportError("The asyncua library is not installed. Please install it and try again.")
