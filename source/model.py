import logging
import operator
import pickle
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import plotly.graph_objs as go
import plotly.offline as plt
from keras.callbacks import Callback, History
from keras.engine.saving import load_model, save_model
from keras.utils import plot_model

import source.config as c
from source.metrics import ModelMetric

events = logging.getLogger(__name__)


class AbstractModel(ABC):
    """ The abstract Model, that all inherit from. """

    def __init__(self, program, feature_set):
        self.feature_set = feature_set
        self.name = str(program)
        self.num_features = len(program.features[feature_set])
        self.epoch = 0
        self.model = {}
        self.scaler = None
        self.path = '{}/models/{}'.format(c.RUN_DIR, repr(self))

    def __getstate__(self) -> dict:
        """ Called when pickling an Agent. """
        state = self.__dict__.copy()
        for name, model in self.model.items():
            self.model[name] = save_model(model, '{}_{}'.format(self.path, name))
            self.model[name] = None
        return state

    def __repr__(self) -> str:
        """ A unique string representation. """
        return "{}_{}".format(str(self.feature_set).split(".")[-1], self.name)

    def __str__(self) -> str:
        """ Non-unique string representation. """
        return self.name

    def save(self) -> None:
        """ Save the provided model to the proper path. """
        with open(self.path, 'wb') as a:
            pickle.dump(self, a)
        return

    @staticmethod
    def load(model_path, recover: bool = True):
        """ Loads a model behind model_path. """
        with open(model_path, 'rb') as a:
            m = pickle.load(a)
        try:
            for name in m.model.keys():
                m.model[name] = load_model('{}_{}'.format(m.path, name))
        except FileNotFoundError:
            if recover:
                events.exception("Model not found, building a new one.", exc_info=True)
                m.build()
            else:
                raise FileNotFoundError("Internal model not found.")
        return m

    @abstractmethod
    def build(self, verbose=0) -> None:
        raise NotImplemented

    @abstractmethod
    def fit(self, training, validation, epochs=100, verbose=0, callbacks=None, **kwargs) -> list:
        """ Trains the agent, on the given programs, for the Agent's feature set.

        :param callbacks:
        :param epochs: Number of epochs to train for.
        :param verbose: How verbose the underlying .fit() methods are.
        :param validation: List of programs to validate against.
        :param training: List of programs to train on.
        """
        raise NotImplemented

    ###############
    # Fit Helpers #
    ###############
    def _fit_submodel(self, key, x, y, validation, epochs=1, batch_size=32, shuffle=True, verbose=0):
        hist = self.model[key].fit(
            x, y, epochs=(self.epoch * epochs) + epochs,
            verbose=verbose, validation_data=validation,
            initial_epoch=self.epoch * epochs, batch_size=batch_size, shuffle=shuffle,
            callbacks=[])
        return hist

    def on_train_begin(self, callbacks):
        for callback in callbacks:
            callback.path = self.path
            callback.model = self.model
        return

    def on_epoch_end(self, callbacks, history):
        for callback in callbacks:
            callback.on_epoch_end(self.epoch, history)
        return

    @abstractmethod
    def evaluate(self, programs, verbose=0, **kwargs) -> list:
        """ Tests the agent, on the given programs, for the Agent's feature set.

        :param verbose:
        :param programs: List of programs to test against.
        """
        raise NotImplemented

    #############
    # Recording #
    #############
    def plot_model(self, key: str) -> None:
        """ Plots model `key`. """
        plot_model(self.model[key], '{}/{}.png'.format(c.RUN_DIR, key), show_shapes=True, rankdir='LR')
        return

    def record_model_weights(self, model_name: str) -> np.array:
        """ Returns an np array of the weights. """
        weights = []
        for layer in self.model[model_name].layers:
            if layer.get_weights():
                weights.extend(layer.get_weights()[0].flatten().tolist())
        return np.array(weights)

    def hist_to_model_metrics(self, hist: History, metric: str, hist_key: str = "loss") -> list:
        """ Converts History objects to collections of ModelMetrics for storing in the DB. """
        return [
            ModelMetric(
                feature_set=self.feature_set,
                name=str(self),
                epoch=e,
                metric=metric,
                value=v,
            ) for e, v in zip(hist.epoch, hist.history[hist_key])
        ]

    #############
    # Callbacks #
    #############
    class PlotRuntimes(Callback):
        def __init__(self, x: dict, y: dict, title: str):
            super().__init__()
            self.x = x
            self.y = y
            self.title = title
            self.auto_open = False

        def on_epoch_end(self, epoch, logs=None):
            # Calculate results
            p_runtimes = self.model.predict(self.x).flatten()
            plt.plot(
                {
                    'data': [
                        go.Scatter(x=[i for i in range(len(p_runtimes))], y=p_runtimes, name="Predicted"),
                        go.Scatter(x=[i for i in range(len(self.y['runtimes']))], y=self.y['runtimes'], name="Actual")
                    ],
                    'layout': go.Layout(title=self.title)
                },
                auto_open=self.auto_open,
                filename=c.RUN_DIR + '/' + self.title + '.html'
            )
            self.auto_open = False
            return

    class GANEarlyStopping(Callback):
        def __init__(self, monitor='val_loss', verbose=0, mode='min', patience=10):
            super().__init__()
            self.model = None
            self.path = None
            self.verbose = verbose
            self.memory = deque(maxlen=patience)
            self.stop_training = False
            self.monitor = monitor
            self.best = np.inf if mode == 'min' else -np.inf
            self.compare = operator.lt if mode == 'min' else operator.gt

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            current = logs.get(self.monitor)[-1]

            # Validate
            if current is None:
                raise ValueError("Invalid metric selected for EarlyStopping.")
            else:
                self.memory.append(current)
            if self.verbose > 0:
                print("The current average validation: ", current)
            # Calculate if the model is best / should stop
            if self.compare(current, self.best):
                self.best = current
                for k, m in self.model.items():
                    m.save("{}_{}".format(self.path, k))

            elif self.best not in self.memory:
                if self.verbose > 0:
                    print("Stopping training at epoch: ", epoch, " accuracy: ", self.best)
                self.stop_training = True
                for k, _ in self.model.items():
                    # TODO: This breaks the model: https://github.com/keras-team/keras/issues/10806, can't train after
                    self.model[k] = load_model("{}_{}".format(self.path, k))
                    # self.model[backend].summary()  # proves it
            return
