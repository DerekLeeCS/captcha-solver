import tensorflow as tf
from matplotlib import pyplot as plt

from dataset.captcha_dataset import CaptchaDataset
from dataset.tfrecord_handler import TFRecordHandler, load_image
from model import create_model, prepare_example, preprocess_image

# From:
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    1e-3,
    decay_steps=5000,
    decay_rate=0.9,
    staircase=True,
)
AUTOTUNE = tf.data.experimental.AUTOTUNE
OPTIMIZER = tf.keras.optimizers.Adam(lr_schedule)
LOSS = tf.keras.losses.CategoricalCrossentropy()
METRIC = tf.keras.metrics.CategoricalAccuracy()

NUM_EPOCHS = 100
BATCH_SIZE = 64

CHECKPOINT_PATH = 'checkpoints/captcha_solver/'
SAVED_MODEL_PATH = 'saved_model/captcha_solver/'


def train_model(model: tf.keras.Model, train_ds: tf.data.Dataset, valid_ds: tf.data.Dataset) -> None:
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        monitor='val_categorical_accuracy',
        verbose=1,
        save_weights_only=True,
        period=20,  # Saves every 20 epochs
        save_best_only=True)

    model.fit(train_ds, epochs=NUM_EPOCHS, validation_data=valid_ds, callbacks=cp_callback)


if __name__ == '__main__':
    img_height, img_width = CaptchaDataset.get_image_height_width()
    num_chars = CaptchaDataset.get_num_chars()
    model = create_model(img_height, img_width, num_chars)
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRIC)
    model.summary()

    train_dataset, valid_dataset, test_dataset = CaptchaDataset.get_data()
    num_train_examples = TFRecordHandler.count_size(train_dataset)

    train_dataset = (
        train_dataset
            .map(prepare_example, num_parallel_calls=AUTOTUNE)
            .cache()
            .shuffle(tf.cast(num_train_examples, tf.int64))
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE)
    )
    valid_dataset = (
        valid_dataset
            .batch(BATCH_SIZE)
            .map(prepare_example, num_parallel_calls=AUTOTUNE)
            .cache()
            .prefetch(AUTOTUNE)
    )
    test_dataset = (
        test_dataset
            .batch(BATCH_SIZE)
            .map(prepare_example, num_parallel_calls=AUTOTUNE)
            .cache()
            .prefetch(AUTOTUNE)
    )

    train_model(model, train_dataset, valid_dataset)
    model.load_weights(CHECKPOINT_PATH)
    model.save(SAVED_MODEL_PATH)

    mappings = CaptchaDataset.get_reverse_label_mappings()
    for batch_img, batch_label in test_dataset.take(1):
        count = 0
        for img, label in zip(batch_img, batch_label):
            plt.imshow(img)
            plt.show()

            label_char = tf.argmax(label, axis=-1)
            label_true = [mappings[tf.get_static_value(x)] for x in label_char]
            print("True:", label_true)
            img = tf.expand_dims(img, axis=0)
            out = model.predict(img)
            out = tf.math.argmax(out, axis=-1)
            out = tf.squeeze(out)
            out_chars = [mappings[tf.get_static_value(x)] for x in out]
            print("Predicted:", out_chars)

            count += 1
            if count == 5:
                break
