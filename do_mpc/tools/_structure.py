from do_mpc.tools import IndexedProperty
import pdb

def _tuplify(f):
    """Decorator ensures input is list.
    """
    def wrapper(self, ind, *args):
        if isinstance(ind, (int,str, slice)):
            ind = (ind,)
        elif isinstance(ind, list):
            ind = tuple(ind)
        return f(self, ind, *args)
    return wrapper

class Structure:
    """ Simple structure class that can hold any type of data.
    Structure is constructed when calling __setitem__ and can grow in complexity.

    **Example:**

    ::

        s = Structure()
        s['_x', 'C_a'] = {'C_a_0':[1,2,3], 'C_a_1': [2,3,4]}
        s['_x', 'C_b'] = 'C_b'
        s['_u', 'C_a'] = 'C_a'

    investigate the indices with ``s.powerindex``. This yield the following:

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
        # List of all stored data.
        self.master = []
        # List of tuples that hold the powerindices for the stored data.
        self.powerindex = []
        # List of possible features for each index.
        self.features = []
        # index:
        self.index = []
        self.count = 0

    @property
    def full(self):
        """Return all elements of the structure.
        Elements are returned in an unnested list.
        """
        return self.master

    @IndexedProperty
    def get_index(self,ind):
        """Get regular indices ([0,1,2, ... N]) for the queried elements.
        This call mimics the __getitem__ method but returns the indices of
        the queried elements instead of their values.

        This is an IndexedProperty and can thus be queried as shown below:

        **Example:**

        ::

            # Sample structure:
            s = Structure()
            s['_x', 'C_a'] = {'C_a_0':[1,2,3], 'C_a_1': [2,3,4]}
            s['_x', 'C_b'] = 'C_b'
            s['_u', 'C_a'] = 'C_a'

            # Get indices:
            s.get_index['_x', 'C_a']
            s.get_index['_x', 'C_a', :, 1:]



        The same nested list structure is obtained when using slices.
        """
        _iter_master, _iter_index  = self._select(ind, self.index, self.powerindex)
        return _iter_master

    @_tuplify
    def __setitem__(self, ind, val):
        if ind in self.powerindex:
            # Reset existing item:
            i = self.powerindex.index(ind)
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
                # Add powerindex
                self.powerindex.append(ind)
                # Add index
                self.index.append(self.count)
                self.count += 1
                # List index as new feature if it does not exist.
                for i,ind_i in enumerate(ind):
                    if len(self.features)<=i:
                        self.features.append([])
                    if ind_i not in self.features[i]:
                        self.features[i].append(ind_i)


    def __getitem__(self, ind):
        _iter_master, _iter_index  = self._select(ind, self.master, self.powerindex)
        return _iter_master

    def _getkeys(self, ind):
        _keys, _  = self._select(ind, self.powerindex, self.powerindex)
        return _keys

    @_tuplify
    def _select(self, ind, _iter_master, _iter_index):
        """Private method to support the __getitem__ call.
        Necessary helper function because it can be called recursively.
        This is required for slicing.
        """
        for j, ind_j in enumerate(ind):
            # temporary candidates for the kept features / data:
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

                # Write temporary candidates to current candidates:
                _iter_master = _tmp_master
                _iter_index = _tmp_index

        return _iter_master, _iter_index



def test_structure():
    s = Structure()
    for k in range(3):
        s['_x', 'C_a', k] = [1,2,3,4]
        s['_x', 'C_b', k] = [1,2,3,4]
        s['_u', 'C_c', k] = [1,2,3,4]


    s['_x',:]


    return s



if __name__ == '__main__':
    s = test_structure()