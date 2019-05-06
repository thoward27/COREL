import logging
from os.path import join
from random import shuffle

from torch import save, cuda

from Benchmarks import Programs
from torch.nn.functional import softmax
from PyTorchRL.agents.AlphaZero import *
from source.MCTS import mcts
from source.config import FLAGS

STEPS = 10


class AlphaCompile(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = len(FLAGS) + 1):
        super().__init__()

        hidden = (input_dim + output_dim) // 2
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.actions = nn.Linear(hidden, output_dim)
        self.value = nn.Linear(hidden, 1)

        self.optim = optim.RMSprop(self.parameters())
        self.loss_value = nn.MSELoss()
        self.loss_policy = nn.BCEWithLogitsLoss()

        self.device = torch.device('cuda' if cuda.is_available() else 'cpu')
        self.to(self.device)
        return

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        h = self.body(softmax(x.to(self.device), dim=0))
        return softmax(self.actions(h), dim=0), tanh(self.value(h))

    def play(self, program, steps=10, render=False):
        logging.debug("Playing {}".format(program))
        state = program.reset()
        for s in range(steps):
            if render:
                program.render()

            if self.training:
                # Policy Loss
                pi = mcts(self, state, deepcopy(program), simulations=2)
                policy, v = self.forward(state)
                loss_policy = self.loss_policy(policy, softmax(torch.tensor([n.w for n in pi]).to(self.device), dim=0))

                # Value Loss
                state, rew, done, info = program.step(max(pi).action)
                loss_value = self.loss_value(v, torch.tensor(rew).to(self.device))

                # Optimizer Step
                loss = loss_policy + loss_value
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # MLFlow
                logging.debug("Action {} got reward {}".format(max(pi).action, round(rew, 2)))
                mlflow.log_metric("Loss", (loss_policy + loss_value).item())
                mlflow.log_metric(str(program), rew)
                mlflow.log_metric("Training Value Error", loss_value.item())
                mlflow.log_metric("Training Policy Error", loss_policy.item())
            else:
                # Calculate best move
                pi, z = self.forward(state)
                state, rew, done, info = program.step(torch.argmax(pi).item())

                # MLFlow
                mlflow.log_metric("Testing {}".format(str(program)), rew)

            if done:
                break


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    programs = Programs()
    train, test = programs.filter(programs[0])
    model = AlphaCompile(train[0].observation_space, len(FLAGS) + 1)
    logging.info(model)

    model.train()
    for e in range(10):
        shuffle(train)
        for program in train:
            try:
                model.play(program)
            except Exception as e:
                logging.exception("{}: {}".format(repr(program), e), exc_info=True)
            finally:
                save(model.state_dict(), join('source', 'AlphaCompile.pth'))

    model.eval()
    for program in test:
        try:
            model.play(program)
        except Exception as e:
            logging.exception("{}: {}".format(repr(program), e), exc_info=True)
