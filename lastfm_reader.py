import os
import collections

import tensorflow as tf


def lastfm_raw_data(data_path =None):
    lastfm_path = os.path.join(data_path, 'userid-timestamp-artid-artname-traid-traname.tsv')

def _generate_ids(path):
    with tf.gfile.GFile(path, "r") as file:
