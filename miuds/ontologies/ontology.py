from collections import namedtuple
from pandas import DataFrame as _PdDataFrame
import numpy as np
from .database import Database


class DataFrame:
    """Wrapper class of pandas.DataFrame
    Unwrap items to arrays, and only provide necessary interfaces of DataFrame.
    """
    def __init__(self, *args, **kwargs):
        self.df = _PdDataFrame(*args, **kwargs)

    def __getitem__(self, key):
        """Return unique values of specific key"""
        values = self.df[key].values
        values = np.unique(values)
        return values

    def __repr__(self):
        # _PdDataFrame.__repr__ is too hard to read.
        # return self.df.__repr__()
        return 'DataFrame(n_keys={}, size={})'.format(
                len(self.df.columns),
                len(self.df))

    def __len__(self):
        return len(self.df)


class Ontology(object):
    def __init__(self, slot_set, database):
        self.slot_set = slot_set
        if isinstance(database, Database):
            self.database = database
        elif isinstance(database, str):
            self.database = Database(database)
        else:
            raise TypeError(database)

    def retrieve(self, constraint):
        """Retrieve database results"""
        results = self.database.select_from_all(self.slot_set, constraint)
        return DataFrame(results)
