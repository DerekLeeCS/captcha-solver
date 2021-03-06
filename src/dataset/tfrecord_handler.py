from pathlib import Path
from typing import List, Union

import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_png_image(filename: str) -> tf.Tensor:
    """Read an image given the file name.
    From:
    https://www.tensorflow.org/api_docs/python/tf/io/read_file
    """
    raw = tf.io.read_file(filename)
    return tf.image.decode_png(raw, channels=3)


class TFRecordHandler:
    def __init__(self, num_chars: int) -> None:
        self.num_chars = num_chars

    @staticmethod
    def _bytes_feature(value):
        """Return a bytes_list from a string / byte.
        From:
        https://www.tensorflow.org/tutorials/load_data/tfrecord
        """
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Return a float_list from a float / double.
        From:
        https://www.tensorflow.org/tutorials/load_data/tfrecord
        """
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Return an int64_list from a bool / enum / int / uint.
        From:
        https://www.tensorflow.org/tutorials/load_data/tfrecord
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def serialize_example(image: np.array, label: List[int]) -> str:
        """Create a tf.train.Example message ready to be written to a file.
        From:
        https://www.tensorflow.org/tutorials/load_data/tfrecord
        """
        # Create a dictionary mapping the feature name to the tf.train.Example-compatible
        # data type.
        feature = {
            'image_raw': TFRecordHandler._bytes_feature(tf.io.serialize_tensor(image)),  # Serialize array to string
            'label_raw': TFRecordHandler._bytes_feature(tf.io.serialize_tensor(label)),
        }

        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    @staticmethod
    def write_examples(file_name: Path, image: List[tf.Tensor], label: List[List[int]]):
        with tf.io.TFRecordWriter(str(file_name)) as writer:
            for img, label in zip(image, label):
                serialized = TFRecordHandler.serialize_example(img, label)
                writer.write(serialized)

    @staticmethod
    def _parse_tfr_element(element):
        """Parse a single example from a TFRecord.
        Based on:
        https://stackoverflow.com/a/60283571 and https://www.tensorflow.org/tutorials/load_data/tfrecord
        """
        feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label_raw': tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(element, feature_description)

        # Process the image to an array
        image_raw = example.pop('image_raw')  # Get byte string
        example['image'] = tf.io.parse_tensor(image_raw, out_type=tf.uint8)  # Restore the array from byte string

        label_raw = example.pop('label_raw')  # Get byte string
        example['label'] = tf.io.parse_tensor(label_raw, out_type=tf.int32)  # Restore the array from byte string

        # Restore the shape of the image
        example['image'] = tf.ensure_shape(example['image'], (None, None, None))
        example['label'] = tf.ensure_shape(example['label'], (None,))

        return example

    @staticmethod
    def read_examples(filename: Union[str, List[str]]) -> tf.data.TFRecordDataset:
        """
        Based on:
        https://www.tensorflow.org/tutorials/load_data/tfrecord
        """
        raw_dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=AUTOTUNE)
        parsed_dataset = raw_dataset.map(TFRecordHandler._parse_tfr_element, num_parallel_calls=AUTOTUNE)
        return parsed_dataset

    @staticmethod
    def count_size(dataset: tf.data.TFRecordDataset) -> int:
        count = 0
        for _ in dataset:
            count += 1
        return count
