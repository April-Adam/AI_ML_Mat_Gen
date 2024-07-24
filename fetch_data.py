import numpy as np
import tensorflow as tf
import tarfile
import urllib.request
import os.path

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float(x):
    try:
        return float(x)
    except:
        return 0

def qm9_data_parse(record):
    features = {
        "N": tf.io.FixedLenFeature([], tf.int64),
        "labels": tf.io.FixedLenFeature([16], tf.float32),
        "elements": tf.io.VarLenFeature(tf.int64),
        "coords": tf.io.VarLenFeature(tf.float32),
    }
    parsed_features = tf.io.parse_single_example(serialized=record, features=features)

    # the coords were flattened before being saved, so it's a 1D array right now.
    # we reshape to dimensions [-1, 4] so that consecutive groups of 4 numbers are grouped.
    # i.e. [A, B, C, D, E, F, G, H, ...] -> [[A, B, C, D], [E, F, G, H], ...] after reshape to [-1, 4].
    # this effectively restores the dimensions of the coords before the flatten.
    coords = tf.reshape(
        tf.sparse.to_dense(parsed_features["coords"], default_value=0), [-1, 4]
    )
    
    elements = tf.sparse.to_dense(parsed_features["elements"], default_value=0) # reconstructs elements tensor from sparse tensor
    return (elements, coords), parsed_features["labels"]


def qm9_prepare_records(lines):
    # atomic numbers of associated elements
    pt = {"C": 2, "H": 1, "O": 8, "N": 7, "F": 9}

    # number of elements
    N = int(lines[0])

    # the ground truth labels are stored in lines[1].
    # lines[1] is of the form: "gbd A\tB\tC\t...", where "A", "B", "C", ... are the ground truth labels (16 of them).
    # lines[1].split("gdb")[1] results in " A\tB\tC..."
    # " A\tB\tC...".split() returns ["A", "B", "C", ...] (str.split with no arguments splits according to whitespace and removes leading and trailing whitespace)
    # "A", "B", "C", ... are all converted to floats
    labels = [float(x) for x in lines[1].split("gdb")[1].split()]

    # each entry from index 2 to index N+1 is of the form "E\tX\tY\tZ\tA", where "E" is one of the elements in `pt`,
    # "X", "Y", "Z" are the x, y, z coordinates respectively of element "E", and "A" is another number associated with "E".
    # thus, x.split()[0] will get the element "E"
    # pt[E] then gets the atomic number of the element
    elements = [pt[x.split()[0]] for x in lines[2 : N + 2]]

    coords = np.empty((N, 4), dtype=np.float64)
    for i in range(N):
        # x.split()[1:] gets and parses the coordinates "X", "Y", "Z", and the last number "A"
        coords[i] = [_float(x) for x in lines[i + 2].split()[1:]]
    
    feature = {
        "N": tf.train.Feature(int64_list=tf.train.Int64List(value=[N])),
        "labels": tf.train.Feature(float_list=tf.train.FloatList(value=labels)),
        "elements": tf.train.Feature(int64_list=tf.train.Int64List(value=elements)),
        "coords": tf.train.Feature(
            float_list=tf.train.FloatList(value=coords.flatten())
        ),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def qm9_fetch():
    raw_filepath = "qm9.tar.bz2"
    record_file = "qm9.tfrecords"

    tar = tarfile.open(raw_filepath, "r:bz2")

    # record file found -> already extracted -> return early
    if os.path.isfile(record_file):
        print("Found existing record file, delete if you want to re-fetch")
        return record_file

    if not os.path.isfile(raw_filepath):
        print("Downloading qm9 data...", end="")
        urllib.request.urlretrieve(
            "https://ndownloader.figshare.com/files/3195389", raw_filepath
        )
        print("File downloaded")
    else:
        print(f"Found downloaded file {raw_filepath}, delete if you want to redownload")

    print("")
    with tf.io.TFRecordWriter(
        record_file, options=tf.io.TFRecordOptions(compression_type="GZIP")
    ) as writer:
        for i in range(1, 133886):
            # print percentage completion of parsing every 100 parses
            if i % 100 == 0:
                print("\r {:.2%}".format(i / 133886), end="")
            
            with tar.extractfile(f"dsgdb9nsd_{i:06d}.xyz") as f:
                lines = [l.decode("UTF-8") for l in f.readlines()] # decode everything to UTF-8
                try:
                    writer.write(qm9_prepare_records(lines).SerializeToString()) # write the record to file
                except ValueError as e:
                    print(i)
                    raise e
    print("")
    return record_file


def qm9_parse(record_file):
    return tf.data.TFRecordDataset(record_file, compression_type="GZIP").map(
        qm9_data_parse
    )