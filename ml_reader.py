import os
import collections
import unittest

import numpy as np

from operator import itemgetter


# TODO: Differentiate between same song name by different artists
def raw_data(data_path=None):
    train_path = os.path.join(data_path, 'ml_train.dat')
    val_path = os.path.join(data_path, 'ml_val.dat')
    test_path = os.path.join(data_path, 'ml_test.dat')

    train_movie_ids = _read_movie_ids(train_path)
    val_movie_ids = _read_movie_ids(val_path)
    test_movie_ids = _read_movie_ids(test_path)

    return train_movie_ids, val_movie_ids, test_movie_ids, len(train_movie_ids)


def context_data(data_path=None):
    data_path = os.path.join(data_path, 'u.item')

    with open(data_path) as f:
        data = f.read().split("\n")
        genres = {element.split("|")[0]: [int(e) for e in element.split("|")[6:24]] for element in data}
        genres.pop('')

        return genres


def _read_movie_ids(path):
    with open(path) as f:
        raw_data = f.read().split("\n")
        data = [element.split("\t") for element in raw_data]
        data.remove([''])
        sorted_data = sorted(data, key=itemgetter(0, 3))

        return [element[1] for element in sorted_data]


def data_iterator(input_data, batch_size, num_steps):
    input_data = np.array(input_data, dtype=np.int32)

    data_len = len(input_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = input_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)


def main():
    raw_data("data/ml-100k")


class LSTMTest(unittest.TestCase):
    def test_lstm(self):
        main()

if __name__ == "__main__":
    unittest.main()
