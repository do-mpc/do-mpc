import pdb

def tuplify(f):
    """Decorator ensures input is list.
    """
    def wrapper(self, ind, *args):
        if isinstance(ind, (int,str)):
            ind = (ind)
        elif isinstance(ind, list):
            ind = tuple(ind)
        return f(self, ind, *args)
    return wrapper

class Structure:
    """ Simple structure class that can hold any type of data.
    Structure is constructed when calling __setitem__ and can grow in complexity.

    Example:

    ::

        s = Structure()
        s['_x', 'C_a'] = {'C_a_0':[1,2,3], 'C_a_1': [2,3,4]}
        s['_x', 'C_b'] = 'C_b'
        s['_u', 'C_a'] = 'C_a'

    investigate the indices with ``s.index``. This yield the following:

    ::

        [('_x', 'C_a', 'C_a_0', 0),
         ('_x', 'C_a', 'C_a_0', 1),
         ('_x', 'C_a', 'C_a_0', 2),
         ('_x', 'C_a', 'C_a_1', 0),
         ('_x', 'C_a', 'C_a_1', 1),
         ('_x', 'C_a', 'C_a_1', 2),
         ('_x', 'C_b'),
         ('_u', 'C_a'),
         ('_x', 'C_a', 'C_a_0', 0),
         ('_x', 'C_a', 'C_a_0', 1),
         ('_x', 'C_a', 'C_a_0', 2),
         ('_x', 'C_a', 'C_a_1'),
         ('_x', 'C_b'),
         ('_u', 'C_a')]

    Query the structure as follows:

    ::

        s['_x', 'C_a']
        >> [1, 2, 3, 2, 3, 4]

        s['_x', 'C_b']
        >> [C_b]

    Slicing is supported:

    ::

        s['_x', 'C_a', :, 1:]
        >> [[[2], [3]], [[3], [4]]]

    and introduces nested lists for each slice element.


    """
    def __init__(self):
        self.master = []
        self.index = []
        self.features = []


    @tuplify
    def __setitem__(self, ind, val):
        if ind in self.index:
            # Reset existing item:
            i = self.index.index(ind)
            self.master[i] = val
        else:
            # Set new item:

            # recursively call __setitem__ if val is a list, tuple or dict
            # this introduces further power indices.
            if isinstance(val, (list, tuple)):
                for i, item_i in enumerate(val):
                    self[ind + (i,)] = item_i
            elif isinstance(val, dict):
                for i, item_i in val.items():
                    self[ind + (i,)] = item_i
            else:
                # Add value to master
                self.master.append(val)
                # Add index
                self.index.append(ind)
                # List index as new feature if it does not exist.
                for i,ind_i in enumerate(ind):
                    if len(self.features)<=i:
                        self.features.append([])
                    if ind_i not in self.features[i]:
                        self.features[i].append(ind_i)


    def __getitem__(self, ind):
        _iter_master, _iter_index  = self._select(ind, self.master, self.index)
        return _iter_master

    @tuplify
    def _select(self, ind, _iter_master, _iter_index):
        """Private method to support the __getitem__ call.
        Necessary helper function because it can be called recursively.
        """
        for j, ind_j in enumerate(ind):
            _tmp_master = []
            _tmp_index = []
            if isinstance(ind_j, slice):
                # Slice case: Slice from features and recursively call _select.
                ind_j = self.features[j][ind_j]
                ind = list(ind)
                for ind_j_k in ind_j:
                    ind[j] = ind_j_k
                    a,b = self._select(ind, _iter_master, _iter_index)
                    _tmp_master.append(a)
                    _tmp_index.append(b)
                _iter_master = _tmp_master
                _iter_index = _tmp_index
                break

            else:
                # Regular case: Iter over all indices and check condition.
                for i, ind_i in enumerate(_iter_index):
                    if j>len(ind_i):
                        None
                    elif ind_i[j] == ind_j:
                        _tmp_master.append(_iter_master[i])
                        _tmp_index.append(_iter_index[i])

                _iter_master = _tmp_master
                _iter_index = _tmp_index

        return _iter_master, _iter_index

class Structure_old:
    """ Simple structure class that can hold any type of data.
    Structure is constructed when calling __setitem__ and can grow in complexity.

    Example:

    ::

        s = Structure()
        s['_x', 'C_a'] = {'C_a_0':[1,2,3], 'C_a_1': 2}
        s['_x', 'C_b'] = 'C_b'
        s['_u', 'C_a'] = 'C_a'

    investigate the indices with ``s.index``. This yield the following:

    ::

        [('_x', 'C_a', 'C_a_0', 0),
         ('_x', 'C_a', 'C_a_0', 1),
         ('_x', 'C_a', 'C_a_0', 2),
         ('_x', 'C_a', 'C_a_1'),
         ('_x', 'C_b'),
         ('_u', 'C_a'),
         ('_x', 'C_a', 'C_a_0', 0),
         ('_x', 'C_a', 'C_a_0', 1),
         ('_x', 'C_a', 'C_a_0', 2),
         ('_x', 'C_a', 'C_a_1'),
         ('_x', 'C_b'),
         ('_u', 'C_a')]

    Query the structure as follows:

    ::

        s['_x', 'C_a']
        >> [1, 2, 3, 2]

        s['C_b']
        >> [C_b]


    """
    def __init__(self):
        self.master = []
        self.index = []

        self.count = 0
        self.counter = []

    def __setitem__(self, ind, val):
        # ensure tuple:
        if not isinstance(ind, tuple):
            ind = tuple([ind])
        # overwrite item if it exists
        try:
            i = self.index.index(ind)
            self.master[i] = val
        except:
            # recursively call __setitem__ if val is a list, tuple or dict
            # this introduces further power indices.
            if isinstance(val, (list, tuple)):
                for i, item_i in enumerate(val):
                    self[ind + (i,)] = item_i
            elif isinstance(val, dict):
                for i, item_i in val.items():
                    self[ind + (i,)] = item_i
            else:
                self.master.append(val)
                self.index.append(ind)
                self.counter.append(self.count)
                self.count += 1


    def __getitem__(self, ind):
        if not isinstance(ind, tuple):
            ind = tuple([ind])
        # Candidate master and index
        _iter_master = self.master
        _iter_index = self.index
        # enumerate and reverse the query index:
        for j, ind_j in enumerate(ind):
            #temporary master and index
            _tmp_master = []
            _tmp_index = []
            _slice_index = []
            for i, ind_i in enumerate(_iter_index):
                if j>len(ind_i):
                    None
                elif ind_j == ind_i[j]:
                    _tmp_master.append(_iter_master[i])
                    _tmp_index.append(_iter_index[i])
                elif isinstance(ind_j, (slice, list, dict, tuple)):
                    raise Exception('Unsupported powerindex of type {}. Note: slicing, list indexing etc. are not supported.'.format(type(ind_j)))

            _iter_master = _tmp_master
            _iter_index = _tmp_index

        return _iter_master
