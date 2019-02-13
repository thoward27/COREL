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
from decimal import Decimal

from keras import optimizers, Input, Model, losses
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

from source.metrics import ProgramMetric
from source.model import *
from source.programs import Programs
from source.utils import action_to_flags


class SingleModel(AbstractModel):

    def __init__(self, program, feature_set):
        super().__init__(program, feature_set)
        self.delta = Decimal(0.1)
        self.delta_decay = Decimal(0.99)
        self.build()

    def fit(self, training, validation, epochs=100, verbose=0, callbacks=None, **kwargs):
        self.scaler = StandardScaler().fit([p.features[self.feature_set] for p in training + validation])

        x_train, y_train = self.generate_data(training)
        x_test, y_test = self.generate_data(validation)
        hist = self.model['stacked'].fit(x_train, y_train, epochs=(self.epoch * epochs) + epochs, verbose=verbose,
                                         validation_data=(x_test, y_test), initial_epoch=self.epoch * epochs,
                                         batch_size=128, shuffle=True, callbacks=[])
        m_metrics = [
            ModelMetric(
                feature_set=self.feature_set,
                name=str(self),
                epoch=e,
                metric='discriminator_loss',
                value=v,
            ) for e, v in zip(hist.epoch, hist.history['loss'])
        ]

        if self.delta >= Decimal('0.000'):
            self.delta = round(self.delta * self.delta_decay, 3)

        self.epoch += 1
        return m_metrics

    def evaluate(self, programs, verbose=0, **kwargs):
        features = np.array(self.scaler.transform([p.features[self.feature_set] for p in programs]))

        output = self.model['stacked'].predict({'features': features})
        p_features = output[0]
        print(np.mean(np.square(np.subtract(features, p_features))))
        p_flags = output[1]
        p_is_optimal = output[2].flatten()

        runtimes = [p.run(f) for p, f in zip(programs, p_flags)]

        results = [
            ProgramMetric(
                feature_set=self.feature_set,
                program=p,
                epoch=self.epoch,
                runtime=r,
                flags=f,
                is_optimal=i,
            ) for p, r, f, i in zip(programs, runtimes, p_flags, p_is_optimal)
        ]

        print('=' * 100)
        [r.print() for r in results]
        return results

    def build(self, verbose=0):
        # Inputs
        features = Input(shape=(self.num_features,), name="features")

        # Optimizer
        opt = optimizers.Adadelta()

        # Model
        g = Dense(52, activation='relu')(features)
        g = Dense(52, activation='relu')(g)
        p_features = Dense(self.num_features, activation='linear', name='p_features')(g)
        g = Dense(52, activation='relu')(g)
        p_flags = Dense(c.N_FLAGS, activation='sigmoid', name='p_flags')(g)
        g = Dense(25, activation='relu')(p_flags)
        p_is_optimal = Dense(1, activation='sigmoid', name='p_is_optimal')(g)

        self.model['stacked'] = Model(inputs=[features], outputs=[p_features, p_flags, p_is_optimal])
        self.model['stacked'].compile(
            loss=losses.logcosh,
            optimizer=opt,
        )
        print(self.model['stacked'].summary())
        # plot_model(self.model['stacked'], c.RUN_DIR + '/model.png', show_shapes=True)
        return

    def is_optimal(self, program, flags) -> int:
        return int(program.run(flags) <= program.optimal_runtime + self.delta)

    def generate_data(self, programs):
        # Features
        features = self.scaler.transform([p.features[self.feature_set] for _ in range(c.N_ACTIONS) for p in programs])

        # Flags
        flags = [action_to_flags(i) for i in range(c.N_ACTIONS) for _ in programs]

        # Is Optimal
        programs = [p for _ in range(c.N_ACTIONS) for p in programs]
        is_optimal = [self.is_optimal(p, f) for p, f in zip(programs, flags)]

        data = zip(features, flags, is_optimal)
        data = list(zip(*[(feat, flag, opt) for feat, flag, opt in data if opt == 1]))

        features = np.array(data[0])
        flags = np.array(data[1])
        is_optimal = np.array(data[2])

        return {'features': features}, {'p_features': features, 'p_flags': flags, 'p_is_optimal': is_optimal}


def main():
    feature_set = c.Features.HYBRID
    programs = Programs()
    programs = programs.filter(programs[0])
    model = SingleModel(programs['testing'][0], feature_set)
    model.fit(programs['training'], programs['validation'], verbose=1)
    model.evaluate(programs['testing'])


if __name__ == "__main__":
    main()
