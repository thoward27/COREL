""" Conditional Tournament Generative Adversarial Network.

Inputs=(noise, features) -> Generator() -> Discriminator() -> Output=(runtime)

Where Generator() is:
    I:(noise, features) -> h1 -> h2 -> O:(7 flags, sigmoid activation)

And Discriminator() is:
    I:(features, flags, flags) -> h1 -> h2 -> O:(1 probability of generator being faster, sigmoid activation)

Loss is logcosh for both.

This model builds on the GAN, adding conditioning to both the generation of flags,
as well as the prediction of runtime. Weight decay is introduced to encourage the
model to use as few flags as possible, while still trying to minimize runtime. This
decay is minimal, yet it seems to greatly help the model escape local minimums.
"""
from collections import defaultdict

from keras import optimizers, Input, Model, losses
from keras.layers import concatenate, Dense, LeakyReLU, regularizers
from sklearn.preprocessing import StandardScaler

import source.utils as u
from source.metrics import ProgramMetric, log
from source.model import *
from source.programs import Programs


class CTGAN(AbstractModel):

    def __init__(self, program, feature_set):
        super().__init__(program, feature_set)
        self.latent_dim = round(self.num_features * 0.1)
        self.build()

    def build(self, verbose=0):
        # Inputs
        noise = Input(shape=(self.latent_dim,), name="noise")
        real_flags = Input(shape=(c.N_FLAGS,), name="real_flags")
        fake_flags = Input(shape=(c.N_FLAGS,), name="fake_flags")
        features = Input(shape=(self.num_features,), name="features")

        # Discriminator
        d = concatenate([features, fake_flags, real_flags])
        d = Dense(int(int(d.shape[1]) / 1.8), kernel_regularizer=regularizers.l2(0.001))(d)
        d = LeakyReLU(0.3)(d)
        d = Dense(int(int(d.shape[1]) / 1.8), kernel_regularizer=regularizers.l2(0.001))(d)
        d = LeakyReLU(0.3)(d)
        d = Dense(int(int(d.shape[1]) / 1.8), kernel_regularizer=regularizers.l2(0.001))(d)
        d = LeakyReLU(0.3)(d)
        d_out = Dense(3, activation='softmax', name='d_out')(d)
        self.model['discriminator'] = Model([features, fake_flags, real_flags], d_out, name="Discriminator")
        self.model['discriminator'].compile(
            loss=losses.binary_crossentropy,
            optimizer=optimizers.Adadelta(),
        )

        # Generator
        g = concatenate([features, noise])
        g = Dense(int(int(d.shape[1]) / 1.8))(g)
        g = LeakyReLU(0.2)(g)
        g = Dense(int(int(d.shape[1]) / 1.8))(g)
        g = LeakyReLU(0.2)(g)
        g_out = Dense(c.N_FLAGS, activation='sigmoid', name='g_out')(g)
        self.model['generator'] = Model([features, noise], g_out, name="Generator")

        # Stacked
        self.model['discriminator'].trainable = False
        faster = self.model['discriminator']([self.model['generator']([features, noise]), features, real_flags])
        self.model['stacked'] = Model([features, noise, real_flags], faster)
        self.model['stacked'].compile(
            loss=losses.binary_crossentropy,
            optimizer=optimizers.Adam(),
        )

        plot_model(self.model['stacked'], c.RUN_DIR + '/model.png', show_shapes=True)
        if verbose > 0:
            print(self.model['stacked'].summary())
        return

    def fit(self, training, validation, epochs=100, verbose=0, callbacks=None, **kwargs):
        callbacks = callbacks or [self.GANEarlyStopping(monitor='wrt_opt', verbose=1, patience=20)]
        self.on_train_begin(callbacks)

        self.scaler = StandardScaler().fit([p.features[self.feature_set] for p in training + validation])

        history = defaultdict(list)
        for _ in range(epochs):
            if any([callback.stop_training for callback in callbacks]):
                break

            # Discriminator
            x_train, y_train = self.generate_discriminator_data(training)
            x_test, y_test = self.generate_discriminator_data(validation)
            hist = self._fit_submodel('discriminator', x_train, y_train, (x_test, y_test), 1, 64, True, verbose)
            history['metrics'].extend(self.hist_to_model_metrics(hist, 'discriminator_loss', 'loss'))

            # Stacked
            x_train, y_train = self.generate_stacked_data(training)
            x_test, y_test = self.generate_stacked_data(validation)
            hist = self._fit_submodel('stacked', x_train, y_train, (x_test, y_test), 1, 32, True, verbose)
            history['metrics'].extend(self.hist_to_model_metrics(hist, 'stacked_loss', 'loss'))

            self.epoch += 1
            history['wrt_opt'].append(np.round(np.mean([float(r.wrt_opt()) for r in self.evaluate(validation)]), 4))

            self.on_epoch_end(callbacks, history)
        return history['metrics']

    def evaluate(self, programs, verbose=0, **kwargs):
        features = self.scaler.transform([p.features[self.feature_set] for p in programs])
        noise = np.random.normal(0, 1, (len(programs), self.latent_dim))
        flags = self.model['generator'].predict({'features': features, 'noise': noise})
        is_optimal = self.model['discriminator'].predict({
            'features': features,
            'real_flags': np.array([u.action_to_flags(0) for _ in programs]),
            'fake_flags': flags
        }).flatten()

        runtimes = [p.run(f) for p, f in zip(programs, flags)]

        results = [
            ProgramMetric(
                feature_set=self.feature_set,
                program=p,
                epoch=self.epoch,
                runtime=r,
                flags=f,
                is_optimal=i,
            ) for p, r, f, i in zip(programs, runtimes, flags, is_optimal)
        ]
        if verbose > 0:
            print('=' * 100)
            [r.print() for r in results]
        return results

    @staticmethod
    def is_faster(program, real, fake) -> list:
        real_runtime = program.run(real)
        fake_runtime = program.run(fake)
        return [int(real_runtime < fake_runtime), int(real_runtime == fake_runtime), int(real_runtime > fake_runtime)]

    def generate_discriminator_data(self, programs):
        features = self.scaler.transform([p.features[self.feature_set] for _ in range(c.N_ACTIONS) for p in programs])

        # Discrete Flags
        real_flags = np.array([u.action_to_flags(i) for i in range(c.N_ACTIONS) for _ in programs])

        fake_flags = np.array(self.model['generator'].predict({
            'noise': np.random.uniform(0, 1, (len(real_flags), self.latent_dim)),
            'features': features,
        }))
        programs = [p for _ in range(c.N_ACTIONS) for p in programs]
        faster = np.array([
            self.is_faster(p, real=r, fake=f) for p, r, f in zip(programs, real_flags, fake_flags)
        ])
        # TODO: Create a function to filter flags to top n%
        return {'real_flags': real_flags, 'fake_flags': fake_flags, 'features': features}, {'d_out': faster}

    def generate_stacked_data(self, programs):
        features = self.scaler.transform(np.array([p.features[self.feature_set] for p in programs]))
        noise = np.random.normal(0, 1, (len(programs), self.latent_dim))
        zeros = np.array([[0, 0, 1] for _ in range(len(programs))])
        real_flags = np.array([u.action_to_flags(i) for i in range(len(programs))])
        return {'noise': noise, 'features': features, 'real_flags': real_flags}, {'Discriminator': zeros}


def main():
    u.init_run()
    programs = Programs()
    for p in programs.names:
        progs = programs.filter(p)
        for f in c.Features:
            model = CTGAN(progs['testing'][0], f)
            log(model.fit(progs['training'], progs['validation'], epochs=1000, verbose=0))
            log(model.evaluate(progs['testing'], verbose=1))


if __name__ == "__main__":
    main()
