from __future__ import division

from reader import Reader
import os
import collections


class LastfmReader(Reader):
    def __init__(self):
        self._data_path = None


    # TODO: Differentiate between same song name by different artists
    def raw_data(self, data_path=None):
        train_path = os.path.join(data_path, 'lastfm_train.dat')
        val_path = os.path.join(data_path, 'lastfm_val.dat')
        test_path = os.path.join(data_path, 'lastfm_test.dat')

        song_ids = LastfmReader._generate_ids(train_path)

        train_data = LastfmReader._file_to_song_ids(train_path, song_ids)
        val_data = LastfmReader._file_to_song_ids(val_path, song_ids)
        test_data = LastfmReader._file_to_song_ids(test_path, song_ids)

        return train_data, val_data, test_data, len(train_data)

    @staticmethod
    def _generate_ids(path):
        song_names = LastfmReader._read_song_names(path)
        counter = collections.Counter(song_names)

        sort = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        songs = [line[0] for line in sort]
        song_to_id = dict(zip(songs, range(len(songs))))

        return song_to_id

    @staticmethod
    def _read_song_names(path):
        with open(path) as f:
            song_names = []
            user = "NONE"
            for line in f:
                split = line.split("\t")
                line_user = split[0]
                if user != line_user:
                    user = line_user
                    #song_names.append("<new>")
                song_names.append(split[5].replace("\n", ""))

            return song_names

    @staticmethod
    def __read_song_names(path):
        with open(path) as f:
            song_names = []
            for line in range(10000):
                line = f.readline()
                song_names.append(line.split("\t")[5].replace("\n", ""))

            return song_names

    @staticmethod
    def _file_to_song_ids(filename, song_to_id):
        song_ids = LastfmReader._read_song_names(filename)

        result = []
        for song in song_ids:
            try:
                result.append(song_to_id[song])
            except KeyError:
                pass

        return result

    @property
    def data_path(self):
        return self._data_path

