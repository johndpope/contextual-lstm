import numpy as np


class Reader(object):
    def raw_data(self, data_path=None):
        raise NotImplementedError("Abstract method")

    def data_iterator(self, input_data, batch_size, num_steps):
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




