from functools import wraps

class IndexedProperty(object):
    """
    Based on the python implementation of the regular property() decorator.
    See for example: https://docs.python.org/3/howto/descriptor.html

    The main tweak is __get__, where the above mentionned implementation
    directly calls the fget function. We instead return the class instance itself,
    where the parent class is now added to the class dict.
    Since the call is followed by brackets, immediatly the __getitem__ or __setitem__
    methods are invoked. These methods are lacking the parent class but it now
    exists in the scope of the property instance.
    We can therefore call fget or fset with the parent class.
    """

    def __init__(self, fget=None, fset=None, doc=None):
        self.fget = fget
        self.fset = fset
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        self.obj = obj
        return self

    def __getitem__(self,ind):
        return self.fget(self.obj,ind)

    def __setitem__(self,ind, val):
        return self.fset(self.obj,ind,val)

    def getter(self, fget):
        @wraps(fget)
        def wrapper(obj,*args, **kwargs):
            return fget(obj, *args, **kwargs)
        self.fget = wrapper
        return type(self)(wrapper, self.fset, self.__doc__)

    def setter(self, fset):
        def wrapper(obj, *args, **kwargs):
            return fset(obj, *args, **kwargs)
        self.fset = wrapper
        return type(self)(self.fget, wrapper, self.__doc__)
