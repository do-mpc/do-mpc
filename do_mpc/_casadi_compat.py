#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
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

"""
Compatibility layer for the casadi 3.8 namespace changes.

casadi 3.8.0 made two backwards incompatible changes that do-mpc relied on:

1. ``casadi.tools`` no longer re-exports the ``casadi`` namespace. Up to casadi
   3.7.2, ``casadi/tools/bounds.py`` started with ``from casadi import *`` and
   ``casadi/tools/__init__.py`` did ``from .bounds import *``, so every casadi
   symbol leaked into ``casadi.tools``. do-mpc's ``castools.SX``,
   ``castools.DM``, ... references depend on that accidental re-export.

2. ``casadi.tools.structure3`` was renamed to ``casadi.tools.structure``.
   Besides the direct references, this also breaks ``pickle.load`` for every
   result file written with an older casadi, because the pickled casadi
   structures carry the module path ``casadi.tools.structure3``.

This module restores the pre-3.8 surface under the name ``castools`` and
registers the old module path, so that existing ``.pkl`` result files stay
loadable. Modules that used to do::

    import casadi.tools as castools

import the facade instead::

    from ._casadi_compat import castools

Note that the reverse direction, reading a pickle written with casadi 3.8 on
casadi <= 3.7, is not recoverable here: casadi's own binary ``Function``
serialization is version gated and refuses to read newer files.
"""
# TODO: The ``castools`` facade is a bit of a hack. It would be cleaner to
#       to rewrite all references that only worked because casadi.tools re-exported the casadi namespace.

import sys as _sys
import casadi.tools as _castools

if hasattr(_castools, 'structure3'):
    # casadi <= 3.7
    structure3 = _castools.structure3
    _sys.modules.setdefault('casadi.tools.structure', structure3)
else:
    # casadi >= 3.8
    structure3 = _castools.structure
    # Keep pickles written with casadi <= 3.7 loadable.
    _sys.modules.setdefault('casadi.tools.structure3', structure3)
    _castools.structure3 = structure3

structure = structure3

# Restore the casadi symbols that casadi.tools used to re-export.
from casadi import *        
from casadi.tools import *  

#: Drop-in replacement for ``import casadi.tools as castools``.
castools = _sys.modules[__name__]


# def __getattr__(name):
#     """Fall back to casadi for anything the star imports did not pick up.

#     This is only reached on a genuine miss, so it costs nothing in normal
#     operation. It matters for the documentation build, where conf.py mocks
#     casadi away via ``autodoc_mock_imports``: without this fallback the
#     import time uses of ``castools.SX`` (a type annotation in Model) and
#     ``castools.struct_SX`` (a base class in tools._casstructure) would raise
#     AttributeError against a real module object.
#     """
#     try:
#         return getattr(_castools, name)
#     except AttributeError:
#         pass
#     import casadi as _casadi
#     try:
#         return getattr(_casadi, name)
#     except AttributeError:
#         raise AttributeError(
#             "module '{}' has no attribute '{}'".format(__name__, name)
#         ) from None
