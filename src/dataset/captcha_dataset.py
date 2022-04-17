import math
import os
import string
from pathlib import Path
from typing import Dict, Tuple

import tensorflow as tf

from dataset.tfrecord_handler import TFRecordHandler, load_image


class CaptchaDataset:
    DIR_TRAIN = Path(__file__, '../data/images/train/')
    DIR_VALID = Path(__file__, '../data/images/valid/')
    DIR_TEST = Path(__file__, '../data/images/test/')

    DIR_TFRECORD_TRAIN = Path(__file__, '../data/tfrecord/train/')
    DIR_TFRECORD_VALID = Path(__file__, '../data/tfrecord/valid/')
    DIR_TFRECORD_TEST = Path(__file__, '../data/tfrecord/test/')

    _NUM_CHARS = 4
    _IMG_HEIGHT = 32
    _IMG_WIDTH = 128

    _LABELS = string.ascii_letters + string.digits
    _LABEL_MAPPINGS = {x: i for i, x in enumerate(_LABELS)}
    _LABEL_REVERSE_MAPPINGS = {v: k for k, v in _LABEL_MAPPINGS.items()}

    @staticmethod
    def read_examples(dir_name: Path) -> Dict:
        examples = {'images': [], 'labels': []}
        count = 0
        for file in os.listdir(dir_name):
            filename = os.path.join(dir_name, file)
            image = load_image(filename)
            label = list(file.split('_')[0])
            label = [CaptchaDataset._LABEL_MAPPINGS[x] for x in label]

            examples['images'].append(image)
            examples['labels'].append(label)
            count += 1
            if count % 1000 == 0:
                print(count)

        return examples

    @staticmethod
    def get_num_chars() -> int:
        return CaptchaDataset._NUM_CHARS

    @staticmethod
    def get_image_height_width() -> Tuple[int, int]:
        return CaptchaDataset._IMG_HEIGHT, CaptchaDataset._IMG_WIDTH

    @staticmethod
    def get_labels() -> str:
        return CaptchaDataset._LABELS

    @staticmethod
    def get_len_label() -> int:
        return len(CaptchaDataset._LABELS)

    @staticmethod
    def get_reverse_label_mappings() -> Dict:
        return CaptchaDataset._LABEL_REVERSE_MAPPINGS

    @staticmethod
    def preprocess_tfrecord(dir_name: Path) -> tf.data.TFRecordDataset:
        """Apply preprocessing to each element in the dataset and cache the results for future use."""

        # Get the absolute path for every TFRecord in the directory
        file_names = [os.path.join(dir_name, file_name) for file_name in os.listdir(dir_name)]

        return TFRecordHandler.read_examples(file_names)

    @staticmethod
    def get_data() -> Tuple[tf.data.TFRecordDataset, tf.data.TFRecordDataset, tf.data.TFRecordDataset]:
        return CaptchaDataset.preprocess_tfrecord(CaptchaDataset.DIR_TFRECORD_TRAIN), \
               CaptchaDataset.preprocess_tfrecord(CaptchaDataset.DIR_TFRECORD_VALID), \
               CaptchaDataset.preprocess_tfrecord(CaptchaDataset.DIR_TFRECORD_TEST)


def write_dataset_to_tfrecord():
    """Split the dataset into training, validation, and test sets. Write each to a TFRecord."""

    def write_split_to_tfrecord(data_dir_name: Path, tfrecord_dir_name: Path, num_files: int):
        """Write a dataset split to a TFRecord.

        :param data_dir_name: the name of the directory that contains the raw data
        :param tfrecord_dir_name: the name of the directory to save the TFRecords to
        :param num_files: the number of files to divide the data into, where each chunk is 100 MB+
        """
        # Create the TFRecord directory if needed
        os.makedirs(tfrecord_dir_name, exist_ok=True)

        data = CaptchaDataset.read_examples(data_dir_name)

        # Calculate the amount of data in each TFRecord
        n = len(data['images'])
        chunk_size = math.ceil(n / num_files)

        # Write the data to a TFRecord in chunks
        count = 1
        for i in range(0, n, chunk_size):
            filename = tfrecord_dir_name / (str(count) + '.tfrecord')
            TFRecordHandler.write_examples(filename, data['images'][i:i + chunk_size],
                                           data['labels'][i:i + chunk_size])
            count += 1

    # Write each split to a TFRecord
    write_split_to_tfrecord(CaptchaDataset.DIR_TRAIN, CaptchaDataset.DIR_TFRECORD_TRAIN, 5)
    write_split_to_tfrecord(CaptchaDataset.DIR_VALID, CaptchaDataset.DIR_TFRECORD_VALID, 1)
    write_split_to_tfrecord(CaptchaDataset.DIR_TEST, CaptchaDataset.DIR_TFRECORD_TEST, 1)


if __name__ == '__main__':
    write_dataset_to_tfrecord()

    # Test reading a TFRecord
    import matplotlib.pyplot as plt

    _, _, ds = CaptchaDataset.get_data()
    for example in ds.take(5):
        img = example.pop('image').numpy()
        plt.imshow(img)
        plt.show()
        print(img)
        print(example)
