from ast import If
import os
from pyexpat import model
from statistics import mode
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# CREATE MODEL
def create_model():
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(units=1, input_shape=[1])]
    )
    model.compile(optimizer="sgd", loss="mean_squared_error")
    return model


def save_model(model):
    model.save("model.h5")
    print("Model saved")


def load_model(filename="model.h5"):
    if os.path.isfile(filename):
        model = tf.keras.models.load_model(filename)
        print("Model loaded")
        return model
    else:
        print("Model not found")
        return None


# CREATE DATASET
def create_dataset():
    x = np.random.rand(100, 1)
    y = 2 * x + 5
    return x, y


if __name__ == "__main__":
    x, y = create_dataset()
    print("x:", x)
    print("y:", y)
    model = load_model()
    if model is None:
        model = create_model()
        model.fit(x, y, epochs=500)
        save_model(model)

    new_x, new_y = create_dataset()
    print("new_x:", new_x)
    print("new_y:", new_y)
    predictions = model.predict(new_x)
    print("predictions:", predictions)
    plt.plot(new_x, new_y, "ro")
    plt.plot(new_x, predictions, "bo")
    plt.show()
    print("Model accuracy:", model.evaluate(new_x, new_y))
