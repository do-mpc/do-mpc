#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2020 Sergio Lucia, Felix Riedl, Alexandru Tatulea-Codrean
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


import time
from threading import Timer

class RealtimeTrigger:
    """
    This class is employed in timing the execution of your real-time ::do-mpc modules. One RealtimeTrigger is required 
    for every module, i.e. one for the simulator, one for the controller and one for the estimator, if the latter is present.
    """
        
    def __init__(self, interval, function, *args, **kwargs):
        """This function implements the server calls and simulator step with a predefined frequency
        
        :param interval: the cycle time in seconds representing the frequency with which the target function is executed
        :type interval: integer
        
        :param function: a function to be called cyclically
        :type function: python function header
        
        :param args: arguments to pass to the target function
        :type args: python dict
        
        :return: none
        :rtype: none
        """
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.next_call = time.time()
        self.start()
    
    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)
  
    def start(self):
        if not self.is_running:
            self.next_call += self.interval
            self._timer = Timer(self.next_call - time.time(), self._run)
            self._timer.start()
            self.is_running = True
  
    def stop(self):
        self._timer.cancel()
        self.is_running = False