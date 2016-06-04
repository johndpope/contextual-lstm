from reader import Reader

# Reader prototype for Lastfm 1k-users data set #


class LastfmReader(Reader):
    def raw_data(self, data_path=None):
        return None

    def data_iterator(self, input_data, batch_size, num_steps):
        return None
