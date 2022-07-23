import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import os
import pandas as pd

labels = ["low", "mid", "high"]
# load data from csv file
def load_data(filename):
    if os.path.isfile(filename):
        data = pd.read_csv(filename)
        # convert string to float (low, mid, high) to number (1, 2, 3) in the last column
        data.iloc[:, -1] = data.iloc[:, -1].map({"low": 0, "mid": 1, "high": 2})
        return data
    else:
        print("File not found")
        return None


# split data into train and test
def split_data(data):
    if data is not None:
        train_data = data.sample(frac=0.8, random_state=0)
        test_data = data.drop(train_data.index)
        y_train = train_data.iloc[:, -1]
        x_train = train_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]
        x_test = test_data.iloc[:, :-1]
        return x_train, y_train, x_test, y_test
    else:
        print("Data not found")
        return None


def create_model():
    inputs = keras.Input(shape=(9,), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(3, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def train_model(model, x_train, y_train, x_val, y_val):
    if (
        model is not None
        and x_train is not None
        and y_train is not None
        and x_val is not None
        and y_val is not None
    ):
        model.fit(
            x_train,
            y_train,
            epochs=100,
            validation_data=(x_val, y_val),
            batch_size=64,
            verbose=2,
        )
        return model
    else:
        print("Model or data not found")
        return None


def evaluate_model(model, x_val, y_val):

    if model is not None and x_val is not None and y_val is not None:
        loss, acc = model.evaluate(x_val, y_val, verbose=2, batch_size=64)
        print("Accuracy: %f" % acc)
        print("Loss: %f" % loss)
        predictions = model.predict(x_val)
        pred_y = []
        for i in predictions:
            pred_y.append(np.argmax(i))
        x = np.array(x_val)
        y = np.array(y_val)
        print("test x:", x)
        print("real y:", y)
        print("predicted y:", pred_y)
        return x, y, pred_y
    else:
        print("Model or data not found")
        return None


def predicte(model, x_val, y_val):

    if model is not None and x_val is not None and y_val is not None:
        x_test = np.array(x_val.iloc[0]).reshape(1, 9)
        y_real = np.array(y_val.iloc[0])
        real_word = labels[y_real]
        print("x test:", x_test)
        predictions = model.predict(x_test)
        print("predicted probabilities:", predictions)
        pred_word = labels[np.argmax(predictions)]
        print("predicted word:", pred_word)
        return x_test, real_word, pred_word
    else:
        print("Model or data not found")
        return None


def plot_model(x, real_y, pred_y):
    plt.plot(x, real_y, "b", label="real")
    plt.plot(x, pred_y, "r", label="pred")
    plt.show()


def save_model(model, filename="EEGmodel.h5"):
    model.save(filename)
    print("Model saved")


def load_model(filename="EEGmodel.h5"):
    if os.path.isfile(filename):
        model = tf.keras.models.load_model(filename)
        return model
    else:
        return None


if __name__ == "__main__":
    data = load_data("BatchTestCSVData0.csv")
    x_train, y_train, x_val, y_val = split_data(data)
    model = load_model()
    if model is None:
        model = create_model()
        model = train_model(model, x_train, y_train, x_val, y_val)
    x, y, pred_y = evaluate_model(model, x_val, y_val)
    predicte(model, x_val, y_val)
    plot_model(x, y, pred_y)
