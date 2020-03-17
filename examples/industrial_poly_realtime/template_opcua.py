#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2020 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

from opcmodules import Server,Client

def template_opcua(model):
    """
    --------------------------------------------------------------------------
    template_opcua: the parameters for the server and client
    --------------------------------------------------------------------------
    """    
    """Defining the settings for the OPCUA server"""
    server_opts = {"_model":model,
                   "_name":"Poly Reactor OPCUA",
                   "_address":"opc.tcp://localhost:4840/freeopcua/server/",
                   "_port": 4840,
                   "_server_type": "basic",
                   "_store_predictions": False,
                   "_with_db": False}
    opc_server = Server(server_opts)

    # The server can only be started if it hasn't been already created 
    if opc_server.created == True and opc_server.running == False: opc_server.start()
    
    
    """Defining the settings for the OPCUA clients, which MUST match the server"""
    client_opts = {"_address":"opc.tcp://localhost:4840/freeopcua/server/",
                   "_port": 4840,
                   "_client_type": "simulator",
                   "_namespace": opc_server.namespace}
    opc_opts ={'_cycle_time': 1.0, '_opc_opts': client_opts}

    return opc_server, opc_opts
