import tensorflow as tf
import matplotlib.pyplot as plt


class LinearModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.W = tf.Variable(5., name='weight')
        self.B = tf.Variable(5., name='bias')

    def call(self, input_data):
        return input_data * self.W + self.B


def loss_function(prediction, truth):
    return tf.reduce_mean((prediction - truth) ** 2)


n_events = 2000
training_inputs = tf.random.normal([n_events])
training_noise = tf.random.normal([n_events])
training_truth = training_inputs*3 + 2 + training_noise


model = LinearModel()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

n_epochs = 300
for i in range(n_epochs):
    with tf.GradientTape() as tape:
        loss = loss_function(model(training_inputs, training=True), training_truth)
    print(f"Epoch {i} loss: {loss}")
    gradients = tape.gradient(loss, [model.W, model.B])
    optimizer.apply_gradients(zip(gradients, [model.W, model.B]))
print(f"Final loss: {loss}")
print(f"W = {model.W.numpy()}, B = {model.B.numpy()}")

model.save_weights("Weights/Linear/linear")

plt.scatter(training_inputs, training_truth)
plt.plot([-5, 5], [-5*model.W+model.B, 5*model.W+model.B], 'r')
plt.show()
