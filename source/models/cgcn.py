""" Conditional Generative Cooperative Network.

Inputs=(noise, features) -> Generator() -> Mediator() -> Output=(runtime)

Where Generator() is:
    I:(noise, features) -> h1 -> h2 -> O:(7 flags, sigmoid activation)

And Discriminator() is:
    I:(features, flags) -> h1 -> h2 -> O:(1 runtime, relu activation)

Loss is logcosh for both.

This model builds on the GCN, adding conditioning to both the generation of flags,
as well as the prediction of runtime. Weight decay is introduced to encourage the
model to use as few flags as possible, while still trying to minimize runtime. This
decay is minimal, yet it seems to greatly help the model escape local minimums.
"""
from collections import defaultdict

from keras import optimizers, Input, Model, losses
from keras.layers import Concatenate, Dense, concatenate, LeakyReLU, regularizers
from sklearn.preprocessing import StandardScaler

from source.metrics import ProgramMetric
from source.model import *
from source.programs import Programs
from source.utils import action_to_flags, init_run


class CGCN(AbstractModel):

    def __init__(self, program, feature_set):
        super().__init__(program, feature_set)
        self.latent_dim = round(self.num_features * 0.1)
        self.build()

    def build(self, verbose=0):
        # Input Features
        input_features = Input(shape=(self.num_features,), name="features")

        # Mediator
        input_flags = Input(shape=(c.N_FLAGS,), name="flags")
        m = concatenate([input_features, input_flags])
        m = Dense((self.num_features + c.N_FLAGS), kernel_regularizer=regularizers.l2(0.01))(m)
        m = LeakyReLU(0.2)(m)
        m = Dense(int(int(m.shape[1]) // 1.6), kernel_regularizer=regularizers.l2(0.01))(m)
        m = LeakyReLU(0.2)(m)
        m = Dense(int(int(m.shape[1]) // 1.6), kernel_regularizer=regularizers.l2(0.01))(m)
        m = LeakyReLU(0.2)(m)
        m = Dense(int(int(m.shape[1]) // 1.6), kernel_regularizer=regularizers.l2(0.01))(m)
        m = LeakyReLU(0.2)(m)
        m = Dense(int(int(m.shape[1]) // 1.6), kernel_regularizer=regularizers.l2(0.01))(m)
        m = LeakyReLU(0.2)(m)
        m_out = Dense(1, activation='linear', name='runtimes')(m)
        self.model['mediator'] = Model([input_flags, input_features], m_out, name="mediator")
        self.model['mediator'].compile(
            loss=losses.logcosh,
            optimizer=optimizers.Adadelta(),
        )

        # Generator
        input_noise = Input(shape=(self.latent_dim,), name="noise")
        g = Concatenate()([input_features, input_noise])
        g = Dense((self.num_features + self.latent_dim), kernel_regularizer=regularizers.l2(0.01))(g)
        g = LeakyReLU(0.2)(g)
        g = Dense(int(int(g.shape[1]) // 1.6), kernel_regularizer=regularizers.l2(0.01))(g)
        g = LeakyReLU(0.2)(g)
        g = Dense(int(int(g.shape[1]) // 1.6), kernel_regularizer=regularizers.l2(0.01))(g)
        g = LeakyReLU(0.2)(g)
        g = Dense(int(int(g.shape[1]) // 1.6), kernel_regularizer=regularizers.l2(0.01))(g)
        g = LeakyReLU(0.2)(g)
        g = Dense(int(int(g.shape[1]) // 1.6), kernel_regularizer=regularizers.l2(0.01))(g)
        g = LeakyReLU(0.2)(g)
        g_out = Dense(c.N_FLAGS, activation='sigmoid', name='g_out')(g)
        self.model['generator'] = Model([input_features, input_noise], g_out, name="generator")

        # Stacked
        self.model['mediator'].trainable = False
        runtime = self.model['mediator']([self.model['generator']([input_features, input_noise]), input_features])
        self.model['stacked'] = Model([input_features, input_noise], runtime)
        self.model['stacked'].compile(
            loss=losses.logcosh,
            optimizer=optimizers.Adadelta(),
        )
        self.plot_model('stacked')

        if verbose > 0:
            print(self.model['stacked'].summary())
        return

    def fit(self, training, validation, epochs=100, verbose=0, callbacks=None, **kwargs):
        callbacks = callbacks or [self.GANEarlyStopping(monitor='wrt_opt', verbose=1)]
        self.on_train_begin(callbacks)

        self.scaler = StandardScaler().fit([p.features[self.feature_set] for p in training + validation])

        history = defaultdict(list)
        for e in range(epochs):
            if any([callback.stop_training for callback in callbacks]):
                break
            x_train, y_train = self.generate_mediator_data(training)
            x_test, y_test = self.generate_mediator_data(validation)
            # TODO: Add PlotRuntimes callbacks
            hist = self._fit_submodel('mediator', x_train, y_train, (x_test, y_test), 1, 128, True, verbose)
            history['metrics'].extend(self.hist_to_model_metrics(hist, "mediator_loss", hist_key='loss'))

            x_train, y_train = self.generate_stacked_data(training)
            x_test, y_test = self.generate_stacked_data(validation)
            hist = self._fit_submodel('stacked', x_train, y_train, (x_test, y_test), 1, 32, True, verbose)
            history['metrics'].extend(self.hist_to_model_metrics(hist, 'stacked_loss', hist_key='loss'))

            self.epoch += 1
            history['wrt_opt'].append(np.round(np.mean([float(r.wrt_opt()) for r in self.evaluate(validation)]), 4))

            self.on_epoch_end(callbacks, history)

        return history['metrics']

    def evaluate(self, programs, verbose=0, **kwargs):
        features = np.array(self.scaler.transform([p.features[self.feature_set] for p in programs]))
        noise = np.random.normal(0, 1, (len(features), self.latent_dim))

        flags = self.model['generator'].predict({'features': features, 'noise': noise})
        is_optimal = self.model['mediator'].predict({'features': features, 'flags': flags}).flatten()

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

    def generate_mediator_data(self, programs):
        features = self.scaler.transform([p.features[self.feature_set] for _ in range(c.N_ACTIONS) for p in programs])

        # Discrete
        flags = [action_to_flags(i) for i in range(c.N_ACTIONS) for _ in programs]

        # Continuous
        flags.extend(self.model['generator'].predict({
            "features": features,
            "noise": np.random.uniform(0, 1, (len(features), self.latent_dim))
        }))

        # Runtimes
        programs = [p for _ in range(c.N_ACTIONS) for p in programs] * 2
        runtimes = [p.run(f) for p, f in zip(programs, flags)]
        return {'features': np.array(list(features) * 2), 'flags': np.array(flags)}, {'runtimes': np.array(runtimes)}

    def generate_stacked_data(self, programs):
        features = self.scaler.transform(np.array([p.features[self.feature_set] for p in programs]))
        noise = np.random.normal(0, 1, (len(programs), self.latent_dim))
        # zeros = np.array([p.optimal_runtime for p in programs])
        zeros = np.zeros((len(programs), 1))
        return {'features': features, 'noise': noise}, {'mediator': zeros}


def main():
    init_run()
    programs = Programs()
    programs = programs.filter(programs[5])
    model = CGCN(programs['testing'][0], c.Features.HYBRID)
    model.fit(training=programs['training'], validation=programs['validation'], verbose=1)
    model.evaluate(programs['testing'], verbose=1)


if __name__ == "__main__":
    main()
