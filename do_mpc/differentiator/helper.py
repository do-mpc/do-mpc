"""
Helper functions for the NLPDifferentiator.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Union, Dict, Tuple, Optional


# Define what is documented in Sphinx
__all__ = ['NLPDifferentiatorSettings', 'NLPDifferentiatorStatus']

@dataclass
class NLPDifferentiatorSettings:
    """Settings for NLPDifferentiator.
    """

    lin_solver: str = field(default_factory = lambda: 'casadi')
    """
    Choose the linear solver for the KKT system. 
    Can be ``'casadi'``, ``'scipy'`` or ``'lstsq'`` (least squares).
    """

    check_LICQ: bool = True
    """
    Check if the constraints are linearly independent at the given optimal solution.    
    The result of this check is stored in :py:class:`NLPDifferentiatorStatus`.

    Warning:
        This feature is computationally demanding and should only be used for debugging purposes.
    """

    check_SC: bool = True
    """
    Check if strict complementarity holds.  
    The result of this check is stored in :py:class:`NLPDifferentiatorStatus`.
    """

    track_residuals: bool = True
    """
    Compute the residuals of the KKT system.
    """

    check_rank: bool = False
    """
    Check if the KKT matrix has full rank.
    The result of this check is stored in :py:class:`NLPDifferentiatorStatus`.
    
    Warning:
        This feature is computationally demanding and should only be used for debugging purposes.
    """

    lstsq_fallback: bool = False
    """
    Fallback to least squares if the linear solver fails.
    """

    active_set_tol : float = 1e-6
    """
    Tolerance for the active set constraints. 
    """

    set_lam_zero: bool = False
    """
    Set the Lagrangen multipliers to exactly zero if they are below the tolerance.
    """

@dataclass
class NLPDifferentiatorStatus:
    """
    Status of the NLPDifferentiator.
    """

    LICQ: Optional[bool] = None
    """
    Linear independence constraint qualification. 
    Status is only updated if ``check_LICQ`` is set to ``True``.
    The value is ``None`` if condition is not checked.
    """

    SC: Optional[bool] = None
    """
    Strict complementarity.
    Status is only updated if ``check_SC`` is set to ``True``.
    The value is ``None`` if condition is not checked.
    """

    residuals: Optional[np.ndarray] = None
    """
    Residuals of the KKT system.
    Status is only updated if ``track_residuals`` is set to ``True``.
    The value is ``None`` if condition is not checked.
    """

    lse_solved: bool = False
    """
    Status of the linear system of equations. ``True`` if the system is solved successfully.
    The value is ``None`` if condition is not checked.
    """

    full_rank: Optional[bool] = None
    """
    Status of the rank of the KKT matrix. ``True`` if the matrix has full rank.
    Status is only updated if ``check_rank`` is set to ``True``.
    The value is ``None`` if condition is not checked.
    """

    sym_KKT: bool = False
    """
    Status of preparing the symbolic KKT matrix. ``True`` if the matrix has been prepared.
    """

    reduced_nlp: bool = False
    """
    Status of preparing the reduced NLP. ``True`` if the NLP has been prepared.
    """


