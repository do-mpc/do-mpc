import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Union, Dict, Tuple, Optional



@dataclass
class NLPDifferentiatorSettings:
    """Settings for NLPDifferentiator.
    """

    lin_solver: str = field(default_factory = lambda: 'casadi')
    """
    Choose the linear solver for the KKT system.
    """

    check_LICQ: bool = True
    """
    Check if the constraints are linearly independent.    

    Warning:
        This feature is computationally demanding and should only be used for debugging purposes.
    """

    check_SC: bool = True
    """
    Check if strict complementarity holds.   
    """

    track_residuals: bool = True
    """
    Compute the residuals of the KKT system.
    """

    check_rank: bool = False
    """

    
    Warning:
        This feature is computationally demanding and should only be used for debugging purposes.
    """

    lstsq_fallback: bool = False
    """
    ...
    """

    active_set_tol : float = 1e-6
    """
    ...
    """

    set_lam_zero: bool = False
    """
    ...
    """

@dataclass
class NLPDifferentiatorStatus:
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


