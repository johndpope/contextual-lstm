class Reader(object):
    def raw_data(self, data_path=None):
        raise NotImplementedError("Abstract method")

    def data_iterator(self, input_data, batch_size, num_steps):
        raise NotImplementedError("Abstract method")



