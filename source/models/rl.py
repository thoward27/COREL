""" Reinforcement Learning.

Inputs=(program features) -> Model() -> Outputs=(array of predicted runtimes, one for each action)

Where Model() is:
    I -> h1 -> h2 -> O

Loss is MSE

The best action is taken as argmin(Outputs), this provides the ability to test one-shot and five-
shot testing natively.
"""
import random

from keras import Sequential, optimizers, losses
from keras.layers import Dense

import source.utils as u
from source.metrics import log
from source.models._model import *
from source.programs import Programs


class RL(AbstractModel):
    """ Reinforcement Learning.

    Inputs=(program features) -> Model() -> Outputs=(array of predicted runtimes, one for each action)

    Where Model() is:
        I -> h1 -> h2 -> O

    Loss is MSE

    The best action is taken as argmin(Outputs), this provides the ability to test one-shot and five-
    shot testing natively.
    """

    def __init__(self, program, feature_set):
        """ Instantiates an agent, given it's target program, and it's feature set.

        :param program -> The target program that the agent will be tested against
        :param feature_set -> The feature set the agent will be exposed to.
        """
        super().__init__(program, feature_set)

        self.memory = deque(maxlen=2000)
        self.state_size = len(program.features[feature_set])
        self.action_size = len(c.ACTIONS)

        self.learning_rate = 0.001

        self.build()

        # Content only for RL model.
        self.batch_size = 50
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.15

    def predict(self, context: np.ndarray, count: int = 1) -> list:
        if self.epsilon >= round(random.random(), 2):
            # Act randomly, explore.
            actions = random.sample(range(c.N_ACTIONS), k=count)
        else:
            # Act intentionally, exploit knowledge.
            actions = self.model['model'].predict(context)
            actions = [a for a in np.argsort(actions[0])[:count]]
        return actions

    def fit(self, training, validation, epochs=100, verbose=0, callbacks=None, **kwargs):
        print("Training RL {}".format(repr(self)))
        self.epsilon = 1
        for e in range(1, epochs + 1):
            program = random.choice(training)
            features = np.array([program.features[self.feature_set]])
            actions = self.predict(features, count=1)

            runtime = program.run(actions[0])

            self.memory.append((features, actions[0], runtime))

            self.epoch += 1

            if self.epoch % 25 == 0:
                batch_size = self.batch_size if self.batch_size < len(self.memory) else len(self.memory) - 5
                mini_batch = random.sample(self.memory, batch_size)

                for state, action, runtime in mini_batch:
                    predicted = self.model['model'].predict(state)
                    predicted[0][action] = runtime
                    self.model['model'].fit(
                        state, predicted,
                        initial_epoch=self.epoch, epochs=self.epoch + 1,
                        verbose=0
                    )

            if e % 10 == 0:
                wrt_03 = program.o3 / runtime
                print("E: {}; Agent: {}; action: {}; wrt -03 {:6.4f}".format(
                    e, repr(self), actions[0], wrt_03)
                )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1 - self.epsilon_decay)
        return

    def evaluate(self, programs, verbose=0, **kwargs):
        print("Testing against {}".format(programs[0]))
        self.epsilon = 0
        for p in programs:
            predicted_runtimes = self.model['model'].predict(np.array([p.features[self.feature_set]]))
            action = predicted_runtimes.argmin()
            # Compute baseline and optimized times.
            o3 = p.o3

            p_runtime = predicted_runtimes[0][action]
            a_runtime = p.run(action)

            wrt_03 = o3 / a_runtime

            print("Opt {:6.3f}; 03 {:6.3f}; M {:6.3f}; Actual {:6.3f}; w.r.t. -03 {:6.3f}; action {}; opt-a {}".format(
                p.optimal_runtime, p.o3, p_runtime, a_runtime, wrt_03, action, p.runtimes
            ))

    def build(self, verbose=0):
        # Compute the average between input and output, as a baseline number of neurons.
        hidden_neurons = int((self.state_size + self.action_size) / 2)
        # Build the model.
        model = Sequential()
        model.add(Dense(hidden_neurons, input_dim=self.state_size, activation='relu', name='features'))
        model.add(Dense(hidden_neurons, activation='relu'))
        model.add(Dense(self.action_size, activation='linear', name='flags'))
        model.compile(
            loss=losses.mse,
            optimizer=optimizers.Adam(lr=self.learning_rate),
        )
        self.model = {'model': model}
        self.plot_model('model')
        return


def main():
    u.init_run()
    programs = Programs()
    programs = programs.filter(programs[5], validation_split=0.0)
    agent = RL(programs['testing'][0], c.Features.DYNAMIC)
    agent.fit(programs['training'], programs['validation'], epochs=10000)
    log(agent.evaluate(programs['testing']))


if __name__ == "__main__":
    main()
