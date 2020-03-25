class Structure:
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

    def __setitem__(self, ind, val):
        # ensure tuple:
        if not isinstance(ind, tuple):
            ind = tuple([ind])

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

    def __getitem__(self, ind):
        # ensure tuple:
        if not isinstance(ind, tuple):
            ind = tuple([ind])

        # empty output:
        out = []
        # loop all indices in structure instance:
        for i, ind_i in enumerate(self.index):
            for j, ind_j in enumerate(ind):
                if j>len(ind_i):
                    break
                elif ind_j != ind_i[j]:
                    break
                elif ind_j == ind[-1]:
                    out.append(self.master[i])  

        return out
