import tensorflow as tf
import numpy as np


FEATURE_COLUMNS = [tf.feature_column.numeric_column('x',  shape=[32, 32, 3], dtype=tf.float32)]


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _write_TFRecords(filename, x, y):
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(x)):
        feature = {'x': _bytes_feature(x[i].tostring()),
                   'y': _int64_feature(y[i])}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()


def prepare_TFRecordDatasets(train_filepath, test_filepath):
    cifar10 = tf.keras.datasets.cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = cifar10
    x_train = (x_train / 255.).astype(np.float32)
    x_test = (x_test / 255.).astype(np.float32)

    _write_TFRecords(train_filepath, x_train, y_train)
    _write_TFRecords(test_filepath, x_test, y_test)


def _parse_record(record):
    keys_to_features = {"x": tf.FixedLenFeature([], tf.string),
                        "y": tf.FixedLenFeature([], tf.int64)}
    parsed = tf.parse_single_example(record, keys_to_features)
    x = tf.decode_raw(parsed["x"], tf.float32)
    y = tf.cast(parsed["y"], tf.int32)
    return {'x': x}, y


def load_dataset(filename, epochs=1, batch_size=32, shuffle=False):
    dataset = tf.data.TFRecordDataset(filenames=[filename])
    if shuffle:
        dataset = dataset.shuffle(100_000)
    dataset = dataset.repeat(epochs)
    dataset = dataset.map(_parse_record)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=2)
    return dataset
