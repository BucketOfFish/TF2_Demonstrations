import tensorflow as tf


class MNIST_Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=[None, None, 1]),
            tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10)
        ])
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.train_dataset, self.test_dataset = self._prepare_mnist_dataloaders()
        print("Set up following MNIST model:")
        self.model.summary()

    def _prepare_mnist_dataloaders(self):
        """
        Normalizes MNIST images, casts data to proper formats, and returns dataloaders.
        Training dataloader is shuffled and set to return batches of size 32.
        Test dataloader is not shuffled, and returns all data in one batch.

        Returns: train_dataset, test_dataset
        """
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        train_images = tf.cast(train_images[..., tf.newaxis]/255, tf.float32)
        test_images = tf.cast(test_images[..., tf.newaxis]/255, tf.float32)
        train_labels = tf.cast(train_labels, tf.int64)
        test_labels = tf.cast(test_labels, tf.int64)

        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(1000).batch(32)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(test_images.shape[0])

        return train_dataset, test_dataset

    def _eval_batch(self, image_batch, label_batch, *, training):
        with tf.GradientTape() as tape:
            prediction_batch = self.model(image_batch, training=training)
            loss_batch = self.loss_function(label_batch, prediction_batch)
        gradients = tape.gradient(loss_batch, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return tf.math.reduce_mean(loss_batch)

    def train(self, *, n_epochs):
        print("Beginning training")
        for i in range(n_epochs):
            for image_batch, label_batch in self.train_dataset:
                train_loss = self._eval_batch(image_batch, label_batch, training=True)
            for image_batch, label_batch in self.test_dataset:
                test_loss = self._eval_batch(image_batch, label_batch, training=False)
            print(f"Epoch {i} trained with training loss of {train_loss:.3f} and test loss of {test_loss:.3f}.")


if __name__ == "__main__":
    mnist_model = MNIST_Model()
    mnist_model.train(n_epochs=10)
