from unittest import TestCase

import numpy as np
from keras import Input, Model, losses, optimizers
from keras.layers import concatenate, Dense

N_LATENT = 100
N_FEATURES = 100
N_FLAGS = 7
N_ROWS = 100


def get_weights(model) -> np.array:
    """ Returns an np array of the weights. """
    weights = []
    for layer in model.layers:
        if layer.get_weights():
            weights.extend(np.round(layer.get_weights()[0].flatten(), 4).tolist())
    return weights


class TestCGAN(TestCase):
    @classmethod
    def setUpClass(cls):
        noise = Input(shape=(N_LATENT,), name="noise")
        flags = Input(shape=(N_FLAGS,), name="flags")
        features = Input(shape=(N_FEATURES,), name="features")

        d = concatenate([features, flags])
        d = Dense(52, activation='relu')(d)
        d = Dense(52, activation='relu')(d)
        d_out = Dense(1, name='d_out')(d)
        cls.D = Model([features, flags], d_out, name="D")
        cls.D.compile(
            loss=losses.binary_crossentropy,
            optimizer=optimizers.Adadelta(),
        )
        cls.D.summary()

        g = concatenate([features, noise])
        g = Dense(52, activation='relu')(g)
        g = Dense(52, activation='relu')(g)
        g_out = Dense(7, activation='sigmoid', name='g_out')(g)
        cls.G = Model([features, noise], g_out, name="G")
        cls.G.summary()

        for l in cls.D.layers:
            l.trainable = False
        gan_out = cls.D([cls.G([features, noise]), features])
        cls.GAN = Model([features, noise], gan_out)
        cls.GAN.compile(
            loss=losses.binary_crossentropy,
            optimizer=optimizers.Adadelta(),
        )
        cls.GAN.summary()
        return

    def test_fit(self):
        _features = np.random.normal(0, 1, (N_ROWS, 100))
        _noise = np.random.normal(0, 1, (N_ROWS, N_LATENT))
        _flags = np.random.uniform(0, 1, (N_ROWS, 7))
        _ones = np.ones((N_ROWS, 1))

        # Proving GAN.fit
        g_weights_before = get_weights(self.G)
        d_weights_before = get_weights(self.D)
        gan_weights_before = get_weights(self.GAN)
        self.GAN.fit([_features, _noise], _ones, epochs=100, verbose=0)
        self.assertEqual(d_weights_before, get_weights(self.D), "Fitting GAN shouldn't change D")
        self.assertNotEqual(g_weights_before, get_weights(self.G), "Fitting GAN should change G")
        self.assertNotEqual(gan_weights_before, get_weights(self.GAN), "Fitting GAN should change GAN")

        # Proving D.fit
        g_weights_before = get_weights(self.G)
        d_weights_before = get_weights(self.D)
        gan_weights_before = get_weights(self.GAN)
        self.D.fit([_features, _flags], _ones, epochs=100, verbose=0)
        self.assertEqual(g_weights_before, get_weights(self.G), "Fitting D shouldn't change G")
        self.assertNotEqual(d_weights_before, get_weights(self.D), "Fitting D should change D")
        self.assertNotEqual(gan_weights_before, get_weights(self.GAN), "Fitting D should change GAN")

        # Proving that we can't fit G alone
        with self.assertRaises(RuntimeError):
            self.G.fit([_features, _noise], _flags)
