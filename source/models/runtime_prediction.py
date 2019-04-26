import plotly as plt
from keras import Input, Model, metrics, losses, optimizers
from keras.layers import concatenate, Dense, regularizers, LeakyReLU
from sklearn.preprocessing import StandardScaler

from metrics import actual, predicted
from source.models._model import *
from source.programs import Programs
from source.utils import action_to_flags


class RuntimePredictor(AbstractModel):

    def __init__(self, program, feature_set):
        super().__init__(program, feature_set)
        self.build()

    def generate_mediator_data(self, programs):
        features = []
        flags = []
        runtimes = []
        for p in programs:
            for i in range(len(c.ACTIONS)):
                features.append(p.features[self.feature_set])
                flags.append(action_to_flags(i))
                runtimes.append(p.run(i))
        features = self.scaler.transform(np.array(features))
        flags = np.array(flags)
        runtimes = np.array(runtimes)
        return features, flags, runtimes

    def fit(self, training, validation, epochs=100, verbose=0, callbacks=None, **kwargs):
        self.scaler = StandardScaler().fit([p.features[self.feature_set] for p in training + validation])
        # Training Data
        features, flags, runtimes = self.generate_mediator_data(training)
        test_features, test_flags, test_runtimes = self.generate_mediator_data(validation)

        # Training / Testing
        for e in range(100000):
            e_per = 20
            self.model['mediator'].fit([flags, features], runtimes, epochs=(e * e_per) + e_per, verbose=1,
                                       validation_data=([test_flags, test_features], test_runtimes),
                                       initial_epoch=e * e_per, batch_size=32, shuffle=True)
            r = self.model['mediator'].evaluate([test_flags, test_features], verbose=0)
            print(list(zip(self.model['mediator'].metrics_names, r)))
            p = self.model['mediator'].predict([test_flags, test_features], verbose=0)

            plt.offline.plot({
                'data': [
                    go.Scatter(x=[i for i in range(len(test_features))], y=p.flatten(), name="Predicted"),
                    go.Scatter(x=[i for i in range(len(test_features))], y=test_runtimes, name="Actual")
                ],
                'layout': go.Layout(title="bananas")
            }, auto_open=False)

    def evaluate(self, programs, verbose=0, **kwargs):
        pass

    def build(self, verbose=0):
        # Inputs
        input_features = Input(shape=(self.num_features,), name="features")
        input_flags = Input(shape=(c.N_FLAGS,), name="flags")

        # Concat
        m = concatenate([input_features, input_flags])

        # Connected layers
        m = Dense(60, kernel_regularizer=regularizers.l2(0.001))(m)
        m = LeakyReLU(0.3)(m)
        m = Dense(30, kernel_regularizer=regularizers.l2(0.001))(m)
        m = LeakyReLU(0.3)(m)
        m = Dense(15, kernel_regularizer=regularizers.l2(0.001))(m)
        m = LeakyReLU(0.3)(m)
        m = Dense(8, kernel_regularizer=regularizers.l2(0.001))(m)
        m = LeakyReLU(0.3)(m)

        # Output
        m_out = Dense(1, activation='relu')(m)
        self.model['mediator'] = Model([input_flags, input_features], m_out)

        self.model['mediator'].compile(
            loss=losses.logcosh,
            optimizer=optimizers.Adam(),
            metrics=[metrics.mae, predicted, actual]
        )
        print(self.model['mediator'].summary())
        return


def main():
    feature_set = c.Features.HYBRID
    programs = Programs()
    programs = programs.filter(programs[0])
    model = RuntimePredictor(programs['testing'][0], feature_set)
    model.fit(programs['training'], programs['validation'])
    model.evaluate(programs)


if __name__ == "__main__":
    main()
