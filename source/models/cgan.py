""" Conditional Generative Adversarial Network.

Inputs=(noise, features) -> Generator() -> Discriminator() -> Output=(runtime)

Where Generator() is:
    I:(noise, features) -> h1 -> h2 -> O:(7 flags, sigmoid activation)

And Discriminator() is:
    I:(features, flags) -> h1 -> h2 -> O:(1 probability of flags being optimal, sigmoid activation)

Loss is logcosh for both.

This model builds on the GAN, adding conditioning to both the generation of flags,
as well as the prediction of runtime. Weight decay is introduced to encourage the
model to use as few flags as possible, while still trying to minimize runtime. This
decay is minimal, yet it seems to greatly help the model escape local minimums.
"""
from collections import defaultdict
from decimal import Decimal

from keras import optimizers, Input, Model, losses
from keras.layers import concatenate, Dense, LeakyReLU, regularizers
from sklearn.preprocessing import StandardScaler

import source.metrics as m
import source.utils as u
from source.model import *
from source.programs import Programs


class CGAN(AbstractModel):

    def __init__(self, program, feature_set):
        super().__init__(program, feature_set)
        self.latent_dim = 10  # round(self.num_features * 0.3)
        self.delta = Decimal(0.1)
        self.delta_decay = Decimal(0.99)
        self.build()
        return

    def build(self, verbose=0):
        # Inputs
        noise = Input(shape=(self.latent_dim,), name="noise")
        flags = Input(shape=(c.N_FLAGS,), name="flags")
        features = Input(shape=(self.num_features,), name="features")

        # Discriminator
        d = concatenate([features, flags])
        d = Dense(int(int(d.shape[1]) // 1.6), kernel_regularizer=regularizers.l2(0.0001))(d)
        d = LeakyReLU(0.2)(d)
        d = Dense(int(int(d.shape[1]) // 1.6), kernel_regularizer=regularizers.l2(0.0001))(d)
        d = LeakyReLU(0.2)(d)
        # d = Dense(int(int(d.shape[1]) // 1.6), kernel_regularizer=regularizers.l2(0.0001))(d)
        # d = LeakyReLU(0.2)(d)
        # d = Dense(int(int(d.shape[1]) // 1.6), kernel_regularizer=regularizers.l2(0.0001))(d)
        # d = LeakyReLU(0.2)(d)
        d_out = Dense(1, activation='sigmoid', name='d_out')(d)
        self.model['discriminator'] = Model([features, flags], d_out, name="Discriminator")
        self.model['discriminator'].compile(
            loss=losses.binary_crossentropy,
            optimizer=optimizers.Adadelta(),
        )

        # Generator
        g = concatenate([features, noise])
        g = Dense(int(int(g.shape[1]) // 1.6), kernel_regularizer=regularizers.l2(0.001))(g)
        g = LeakyReLU(0.2)(g)
        g = Dense(int(int(g.shape[1]) // 1.6), kernel_regularizer=regularizers.l2(0.001))(g)
        g = LeakyReLU(0.2)(g)
        # g = Dense(int(int(g.shape[1]) // 1.6), kernel_regularizer=regularizers.l2(0.001))(g)
        # g = LeakyReLU(0.2)(g)
        # g = Dense(int(int(g.shape[1]) // 1.6), kernel_regularizer=regularizers.l2(0.001))(g)
        # g = LeakyReLU(0.2)(g)
        g_out = Dense(c.N_FLAGS, activation='sigmoid', name='g_output')(g)
        self.model['generator'] = Model([features, noise], g_out, name="Generator")

        # Stacked
        for l in self.model['discriminator'].layers:
            l.trainable = False
        d_out = self.model['discriminator']([self.model['generator']([features, noise]), features])
        self.model['stacked'] = Model([features, noise], d_out)
        self.model['stacked'].compile(
            loss=losses.binary_crossentropy,
            optimizer=optimizers.Adadelta(),
        )
        plot_model(self.model['stacked'], c.RUN_DIR + '/model.png', show_shapes=True)

        if verbose > 0:
            print(self.model['stacked'].summary())
        return

    def fit(self, training, validation, epochs=100, verbose=0, callbacks=None, **kwargs):
        # Setup Callbacks
        callbacks = callbacks or [self.GANEarlyStopping(monitor='one_shot', patience=100, verbose=1)]
        self.on_train_begin(callbacks)

        self.scaler = StandardScaler().fit([p.features[self.feature_set] for p in training + validation])

        history = defaultdict(list)
        for _ in range(epochs):
            if any([callback.stop_training for callback in callbacks]):
                break

            # Fit Discriminator
            x_train, y_train = self.generate_discriminator_data(training)
            x_test, y_test = self.generate_discriminator_data(validation)
            hist = self._fit_submodel('discriminator', x_train, y_train, (x_test, y_test), 1, 128, True, verbose)
            history['metrics'].extend(self.hist_to_model_metrics(hist, "discriminator_loss", hist_key="loss"))

            # Fit Stacked
            x_train, y_train = self.generate_stacked_data(training)
            x_test, y_test = self.generate_stacked_data(validation)
            hist = self._fit_submodel('stacked', x_train, y_train, (x_test, y_test), 2, 32, True, verbose)
            history['metrics'].extend(self.hist_to_model_metrics(hist, "stacked_loss", hist_key='loss'))

            self.epoch += 1
            history['one_shot'].append(np.round(np.mean([float(r.one_shot) for r in self.evaluate(validation, n_tries=1)]), 4))

            for callback in callbacks:
                callback.on_epoch_end(self.epoch, history)

            # Delta Decay
            if self.delta >= Decimal('0.000'):
                self.delta = round(self.delta * self.delta_decay, 3)

        return history['metrics']

    def evaluate(self, programs, verbose=0, n_tries=10, **kwargs):
        features = np.array(self.scaler.transform([p.features[self.feature_set] for p in programs * n_tries]))
        noise = np.random.normal(0, 1, (len(features), self.latent_dim))

        flags = self.model['generator'].predict({'features': features, 'noise': noise})

        runtimes = [p.run(f) for p, f in zip(programs * n_tries, flags)]

        results = [m.ProgramMetric(
            feature_set=self.feature_set,
            program=p,
            epoch=self.epoch,
            one_shot=runtimes[i],
            five_shot=min(runtimes[i:len(programs) * 5:len(programs)]),
            ten_shot=min(runtimes[i:len(programs) * 10:len(programs)]),
        ) for i, p in enumerate(programs)]
        if verbose > 0:
            print('=' * 200)
            [r.print() for r in results]
        return results

    def is_optimal(self, program, flags) -> int:
        return int(program.run(flags) <= program.optimal_runtime + self.delta)

    def generate_discriminator_data(self, programs):
        # Features
        features = self.scaler.transform([p.features[self.feature_set] for _ in range(c.N_ACTIONS) for p in programs])

        # Discrete and Continuous Flags
        flags = [u.action_to_flags(i) for i in range(c.N_ACTIONS) for _ in programs]
        flags.extend(self.model['generator'].predict({
            'noise': np.random.uniform(0, 1, (len(features), self.latent_dim)),
            'features': features,
        }))

        # Is Optimal
        programs = [p for _ in range(c.N_ACTIONS) for p in programs] * 2
        is_optimal = [self.is_optimal(p, f) for p, f in zip(programs, flags)]

        # Fix features to match new length
        return {'flags': np.array(flags), 'features': np.array(list(features) * 2)}, {'d_out': np.array(is_optimal)}

    def generate_stacked_data(self, programs):
        features = self.scaler.transform(np.array([p.features[self.feature_set] for p in programs]))
        noise = np.random.normal(0, 1, (len(programs), self.latent_dim))
        ones = np.ones((len(programs), 1))
        return {'noise': noise, 'features': features}, {'Discriminator': ones}


def main():
    u.init_run()
    programs = Programs()
    for p in programs.names:
        progs = programs.filter(p)
        for f in c.Features:
            model = CGAN(progs['testing'][0], f)
            m.log(model.fit(progs['training'], progs['validation'], epochs=1000, verbose=0))
            m.log(model.evaluate(progs['testing'], verbose=1))


if __name__ == "__main__":
    main()
