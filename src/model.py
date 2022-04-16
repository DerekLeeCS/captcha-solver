from typing import Dict, Tuple

import tensorflow as tf

from dataset.captcha_dataset import CaptchaDataset

OUTPUTS = CaptchaDataset.get_labels()

def _get_layer_name(char_num: int) -> str:
    return f'out_{char_num}'


def create_model(h: int, w: int, n: int) -> tf.keras.Model:
    """
    :param h: height of the input image
    :param w: width of the input image
    :param n: number of output characters
    :return: the constructed model
    """
    inp = tf.keras.Input(shape=(None, None, 3))
    x = inp
    x = tf.keras.layers.Resizing(h, w)(x)

    for _ in range(3):
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(n * len(OUTPUTS))(x)
    x = tf.reshape(x, (-1, n, len(OUTPUTS)))
    out = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    return model


def prepare_example(example: Dict, n: int) -> Tuple[tf.Tensor, tf.Tensor]:
    image = example['image']
    image = tf.image.per_image_standardization(image)

    label = example['label']
    label = tf.one_hot(label, len(OUTPUTS))

    return image, label
