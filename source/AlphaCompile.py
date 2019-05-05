from random import shuffle

from Benchmarks import Programs
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
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.actions = nn.Linear(hidden, output_dim)
        self.value = nn.Linear(hidden, 1)

        self.optim = optim.Adam(self.parameters())
        self.loss_value = nn.MSELoss()
        self.loss_policy = nn.BCEWithLogitsLoss()
        return

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        h = self.body(functional.softmax(x, dim=0))
        return functional.softmax(self.actions(h), dim=0), tanh(self.value(h))

    def play(self, game, episodes=10, steps=10, render=False):
        for e in range(episodes):
            logging.debug("Episode {}: {}".format(e, game))
            state = game.reset()
            for s in range(steps):
                if render:
                    game.render()

                if self.training:
                    # Policy Update
                    pi = mcts(self, state, deepcopy(game), simulations=25)
                    policy, v = self.forward(state)
                    loss_policy = self.loss_policy(policy, functional.softmax(torch.tensor([n.w for n in pi]), dim=0))
                    logging.debug("{} -> w():{}".format([round(n.item(), 2) for n in policy], [round(n.w, 2) for n in pi]))
                    logging.debug("{} -> pr():{}".format([round(n.item(), 2) for n in policy], [round(n.pr(), 2) for n in pi]))
                    logging.debug([round(n.item(), 2) for n in self.forward(state)[0]])

                    # Value Update
                    state, rew, done, info = game.step(max(pi).action)
                    loss_value = self.loss_value(v, torch.tensor(rew))

                    logging.debug("Action {} got reward {}".format(max(pi).action, round(rew, 2)))

                    loss = loss_policy + loss_value
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    # MLFlow
                    mlflow.log_metric("Loss", (loss_policy + loss_value).item())
                    mlflow.log_metric("Training reward", rew)
                    mlflow.log_metric("Training Value Error", loss_value.item())
                    mlflow.log_metric("Training Policy Error", loss_policy.item())
                else:
                    pi, z = self.forward(state)
                    state, rew, done, info = game.step(torch.argmax(pi).item())

                    mlflow.log_metric("Testing reward", rew)
                if done:
                    break


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    programs = Programs()
    train, test = programs.filter(programs[0])
    model = AlphaCompile(train[0].observation_space, len(FLAGS) + 1)
    logging.info(model)
    shuffle(train)
    for program in train:
        logging.info("Training on {}".format(repr(program)))
        model.train()
        model.play(program)

    for program in test:
        logging.info("Testing on {}".format(repr(program)))
        model.eval()
        model.play(program)
