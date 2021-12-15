"""
File responsible for preparing training data, training the model, and predicting future carbon
emissions.
"""

import datetime
import os
import random
from typing import List, Tuple

import numpy as np
import tensorflow
from keras.layers import Dense, LSTM
from keras.models import Sequential, load_model

import data


class Model:
    """Keras model based on the LSTM architecture.

    The model predicts future carbon dioxide emissions based on the past days.
    """
    layers: Sequential

    def __init__(self) -> None:
        """Initialzier for the Model"""
        # We need to disable the gpu because it requires an nvidia gpu as well as extra software
        # (CUDNN)
        tensorflow.config.set_visible_devices([], 'GPU')
        self.layers = Sequential()

    def create(self) -> None:
        """Create the model using the LSTM architecture."""
        self.layers.add(LSTM(units=4, return_sequences=True, input_shape=(1, 1)))
        self.layers.add(LSTM(units=4))
        self.layers.add(Dense(units=1))
        self.layers.compile(loss='mean_squared_error', optimizer='adam')

    def save(self, filename: str) -> None:
        """Save model to file"""
        self.layers.save(filename)

    def load(self, filename: str) -> None:
        """Load model from file"""
        self.layers = load_model(filename)

    def train(self, normalized: bool = False, epochs: int = 300, batch_size: int = 8) -> None:
        """Trains the model based on data from prepare_training_data()"""
        x_train, y_train = prepare_training_data(
            './carbonmonitor-us_datas_2021-11-04.csv',
            normalized=normalized
        )
        x_train = reshape_x_data(x_train)
        self.layers.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, x_data: List[float]) -> List[float]:
        """Predicts future carbon emissions for any given time period's data."""
        x_data = reshape_x_data(x_data)
        prediction = list(self.layers.predict(x_data, verbose=0))
        for i in range(len(prediction)):
            prediction[i] = prediction[i][0]
        return prediction


def ensure_reproducible_results(seed: int = 1326) -> None:
    """Ensures that training results are consistent given the same seed"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)
    session_conf = tensorflow.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )
    sess = tensorflow.compat.v1.Session(
        graph=tensorflow.compat.v1.get_default_graph(),
        config=session_conf
    )
    tensorflow.compat.v1.keras.backend.set_session(sess)


def reshape_x_data(x_data: List[float]) -> List[List[List[float]]]:
    """Reshapes a one dimensional input data to three dimensions.

    The reshape ensures that the LSTM layer of the model receives the correct input size.

    >>> reshape_x_data([1, 2, 3])
    [[[1]], [[2]], [[3]]]
    """
    # ACCUMULATOR new_x_data
    new_x_data = []
    for x in x_data:
        new_x_data.append([[x]])
    return new_x_data


def prepare_training_data(filename: str, normalized: bool = False) -> \
        Tuple[List[float], List[float]]:
    """Creates two lists to be used as training data for the model."""
    data_manager = data.DataManager()
    data_manager.load_data(filename)
    data_manager.filter(before_day=datetime.date(2020, 2, 11))
    data_manager.sum_of_total_emissions('United States')
    if normalized:
        data_manager.normalize_data()

    # ACCUMULATOR x_train: input to the model
    x_train = []
    # ACCUMULATOR y_train: expected output from the model
    y_train = []
    for i in range(len(data_manager) - 1):
        x_train.append(data_manager.metrics[i].value)
        y_train.append(data_manager.metrics[i + 1].value)
    return x_train, y_train


def train_and_save(normalized: bool = False) -> Model:
    """Trains the model then saves it to a standardized name.

    This is a helper function that can be used in the python console to train the model.
    """
    model = Model()
    model.create()
    if normalized:
        model.train(normalized=normalized)
        model.save('normalized_data_model.keras')
    else:
        model.train()
        model.save('model.keras')

    return model


def load_and_predict(x_data: List[float], normalized: bool = False) -> List[float]:
    """Loads model based on standardised names and predict future emissions from given data."""
    model = Model()
    if normalized:
        model.load('normalized_data_model.keras')
    else:
        model.load('model.keras')
    return model.predict(x_data)


def denormalize_data(list_data: List, min_val: float, max_val: float) -> None:
    """Reverses the normalize operation DataManager.normalize_data()

    MUTATES list_data.
    """
    for i in range(len(list_data)):
        list_data[i] = (list_data[i] * (max_val - min_val)) + min_val


def main(normalized: bool = False) -> tuple[list, list]:
    """The main function uses the trained models to come up with a prediction"""
    ensure_reproducible_results()

    dm = data.DataManager()
    dm.load_data('./carbonmonitor-us_datas_2021-11-04.csv')
    dm.filter(before_day=datetime.date(2020, 2, 11), exclude=True)
    dm.sum_of_total_emissions('United States')

    if normalized:
        min_val, max_val = dm.get_min_value(), dm.get_max_value()
        dm.normalize_data()
        actual = [metric.value for metric in dm.metrics]
        prediction = load_and_predict(actual, normalized=True)
        denormalize_data(actual, min_val, max_val)
        denormalize_data(prediction, min_val, max_val)
    else:
        actual = [metric.value for metric in dm.metrics]
        prediction = load_and_predict(actual)

    return actual, prediction


if __name__ == "__main__":
    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['datetime', 'os', 'random', 'typing', 'numpy', 'tensorflow',
                          'keras.layers', 'keras.models', 'data'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    import python_ta.contracts
    python_ta.contracts.DEBUG_CONTRACTS = False
    python_ta.contracts.check_all_contracts()

    import doctest
    doctest.testmod()
