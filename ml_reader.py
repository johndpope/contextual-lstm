import os

from operator import itemgetter
from reader import Reader


class MlReader(Reader):

    def __init__(self):
        self._data_path = None

    def raw_data(self, data_path=None):
        self._data_path = data_path
        train_path = os.path.join(data_path, 'ml_train.dat')
        val_path = os.path.join(data_path, 'ml_val.dat')
        test_path = os.path.join(data_path, 'ml_test.dat')

        train_movie_ids = self._read_movie_ids(train_path)
        val_movie_ids = self._read_movie_ids(val_path)
        test_movie_ids = self._read_movie_ids(test_path)

        return train_movie_ids, val_movie_ids, test_movie_ids, len(train_movie_ids)

    @staticmethod
    def _read_movie_ids(path):
        with open(path) as f:
            raw_data = f.read().split("\n")
            data = [element.split("\t") for element in raw_data]
            data.remove([''])
            sorted_data = sorted(data, key=itemgetter(0, 3))

            return [element[1] for element in sorted_data]

    @property
    def data_path(self):
        return self._data_path
