import tensorflow as tf
import pathlib
import time


class MNIST_Model():
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=[None, None, 1]),
            tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10)
        ])
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.train_batch_size = 32
        self.train_loader, self.test_loader, self.train_steps_per_epoch = self._get_dataloaders(self.train_batch_size)
        print("Set up following MNIST model:")
        self.model.summary()

    def _get_dataloaders(self, train_batch_size):
        """
        Normalizes MNIST images, casts data to proper formats, and returns dataloaders.
        Training dataloader is shuffled and set to return batches of size train_batch_size.
        Test dataloader is not shuffled, and returns all data in one batch.

        Returns: train_dataset, test_dataset, train_steps_per_epoch
        """
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        train_images = tf.cast(train_images[..., tf.newaxis]/255, tf.float32)
        test_images = tf.cast(test_images[..., tf.newaxis]/255, tf.float32)
        train_labels = tf.cast(train_labels, tf.int64)
        test_labels = tf.cast(test_labels, tf.int64)
        train_loader = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_loader = train_loader.repeat().shuffle(1000).batch(train_batch_size)
        test_loader = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        test_loader = test_loader.repeat().batch(test_images.shape[0])
        train_steps_per_epoch = train_images.shape[0]//train_batch_size
        return train_loader, test_loader, train_steps_per_epoch

    def train(self, n_epochs):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function)
        self.model.fit(
            self.train_loader,
            epochs=n_epochs,
            steps_per_epoch=self.train_steps_per_epoch,
            validation_data=self.test_loader,
            validation_steps=1)

    def save(self, weights_path):
        pathlib.Path(weights_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(weights_path)

    def load(self, weights_path):
        self.model.load(weights_path)


if __name__ == "__main__":
    begin_time = time.time()
    mnist_model = MNIST_Model()
    mnist_model.train(2)
    end_time = time.time()
    print("_________________________________________________________________")
    print(f"Finished training in {end_time-begin_time:.1f} seconds.")
    # mnist_model.save("Weights/MNIST/")
