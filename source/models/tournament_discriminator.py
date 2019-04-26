""" A tournament style discriminator prototype. """
import plotly
from keras import losses, optimizers, metrics, regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, concatenate, LeakyReLU
from keras.models import Model
from sklearn.preprocessing import StandardScaler

from source.models._model import *
from source.programs import Programs
from source.utils import action_to_flags, flags_to_action


class TournamentDiscriminator(AbstractModel):

    def __init__(self, program, feature_set):
        super().__init__(program, feature_set)
        self.build()

    def fit(self, training, validation, epochs=100, verbose=0, callbacks=None, **kwargs) -> None:
        self.scaler = StandardScaler().fit([p.features[self.feature_set] for p in training + validation])

        for e in range(epochs):
            x_train, y_train = self.generate_discriminator_data(training)
            x_test, y_test = self.generate_discriminator_data(validation)
            self.model['discriminator'].fit(x_train, y_train, epochs=10000, validation_data=(x_test, y_test),
                                            batch_size=64, callbacks=[
                    EarlyStopping(patience=50),
                ], shuffle=True)

            predicted_y = self.model['discriminator'].predict(x_train)
            plotly.offline.plot({
                'data': [
                    go.Histogram(x=np.argmax(predicted_y, axis=1).flatten(), nbinsx=3,
                                 opacity=0.75, name="Predicted"),
                    go.Histogram(x=np.argmax(y_train['output'], axis=1).flatten(), nbinsx=3,
                                 opacity=0.75, name="Actual")
                ],
                'layout': go.Layout(barmode='overlay', title="Training")
            }, filename="training.html")

            predicted_y = self.model['discriminator'].predict(x_test)
            plotly.offline.plot({
                'data': [
                    go.Histogram(x=np.argmax(predicted_y, axis=1).flatten(), nbinsx=3,
                                 opacity=0.75, name="Predicted"),
                    go.Histogram(x=np.argmax(y_test['output'], axis=1).flatten(), nbinsx=3,
                                 opacity=0.75, name="Actual")
                ],
                'layout': go.Layout(barmode='overlay', title="Testing")
            }, filename="testing.html")

    def evaluate(self, programs, verbose=0, **kwargs):
        pass

    def build(self, verbose=0):
        # Inputs
        real_flags = Input(shape=(c.N_FLAGS,), name="real_flags")
        fake_flags = Input(shape=(c.N_FLAGS,), name="fake_flags")
        features = Input(shape=(self.num_features,), name="features")

        # Build
        d = concatenate([features, fake_flags, real_flags])
        d = Dense(52, kernel_regularizer=regularizers.l2(0.001))(d)
        d = LeakyReLU(0.3)(d)
        d = Dense(52, kernel_regularizer=regularizers.l2(0.001))(d)
        d = LeakyReLU(0.3)(d)
        d = Dense(52, kernel_regularizer=regularizers.l2(0.001))(d)
        d = LeakyReLU(0.3)(d)
        faster = Dense(3, activation='softmax', name='output')(d)
        self.model['discriminator'] = Model([features, fake_flags, real_flags], faster)
        self.model['discriminator'].compile(
            loss=losses.binary_crossentropy,
            optimizer=optimizers.Adadelta(),
            metrics=[metrics.mae]
        )
        print(self.model['discriminator'].summary())
        return

    def generate_discriminator_data(self, programs):
        real_flags = []
        fake_flags = []
        features = []
        faster = []
        for p in programs:
            for i in range(c.N_ACTIONS):
                real_flags.append(action_to_flags(i))
                fake_flags.append(np.random.uniform(0, 1, c.N_FLAGS))
                features.append(p.features[self.feature_set])
                faster.append(self.is_faster(p, i, flags_to_action(fake_flags[-1])))
        real_flags = np.array(real_flags)
        fake_flags = np.array(fake_flags)
        features = self.scaler.transform(np.array(features))
        faster = np.array(faster)
        return {'real_flags': real_flags, 'fake_flags': fake_flags, 'features': features}, {'output': faster}

    @staticmethod
    def is_faster(program, real, fake) -> list:
        real_runtime = program.run(real)
        fake_runtime = program.run(fake)
        return [int(real_runtime < fake_runtime), int(real_runtime == fake_runtime), int(real_runtime > fake_runtime)]


def main():
    feature_set = c.Features.HYBRID
    programs = Programs()
    programs = programs.filter(programs[0])
    model = TournamentDiscriminator(programs['testing'][0], feature_set)
    model.fit(programs['training'], programs['validation'])
    model.evaluate(programs)


if __name__ == "__main__":
    main()
