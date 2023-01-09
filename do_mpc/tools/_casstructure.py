import casadi as cas
import casadi.tools as ctools

class _struct_SX(ctools.struct_SX):
    """Updated structure class for CasADi structures (SX). This class fixes a bug that prevents unpickeling of the structure."""
    def __init__(self, *args, **kwargs):
        kwargs.pop('order', None)
        super().__init__(*args, **kwargs) 

class _struct_MX(ctools.struct_MX):
    def __init__(self, *args, **kwargs):
        kwargs.pop('order', None)
        super().__init__(*args, **kwargs) 

class _SymVar:
    def __init__(self, symvar_type):
        assert symvar_type in ['SX', 'MX'], 'symvar_type must be either SX or MX, you have: {}'.format(symvar_type)

        if symvar_type == 'MX':
            self.sym = cas.MX.sym
            self.struct = _struct_MX
            self.sym_struct = ctools.struct_symMX
            self.dtype = cas.MX
        if symvar_type == 'SX':
            self.sym = cas.SX.sym
            self.struct = _struct_SX
            self.sym_struct = ctools.struct_symSX
            self.dtype = cas.SX