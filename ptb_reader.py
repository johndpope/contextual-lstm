from reader import Reader
from tensorflow.models.rnn.ptb import reader


# Wrapper for the PTB reader provided by Tensorflow #
class PtbReader(Reader):
    def raw_data(self, data_path=None):
        return reader.ptb_raw_data(data_path)

    def data_iterator(self, raw_data, batch_size, num_steps):
        return reader.ptb_iterator(raw_data, batch_size, num_steps)