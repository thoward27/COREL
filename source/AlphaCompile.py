import logging
from copy import deepcopy
from os.path import join
from random import shuffle

import mlflow
import numpy as np
import torch
from torch import save, cuda, jit, randn, nn, optim, tanh
from torch.nn.functional import softmax

from Benchmarks import Programs
from Benchmarks.cBench.cBench import MEAN, STD
from source.MCTS import mcts
from source.config import FLAGS

STEPS = 10
EPOCHS = 1
SIMS = 64


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

        self.optim = optim.Adam(self.parameters())
        self.loss_value = nn.MSELoss()
        self.loss_policy = nn.BCEWithLogitsLoss()

        self.device = torch.device('cuda' if cuda.is_available() else 'cpu')
        self.to(self.device)
        return

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = (x - MEAN) / STD
        x[x != x] = 0  # Removes NaNs
        x[x == float('inf')] = 5
        x[x == float('-inf')] = -5
        h = self.body(x.to(self.device))
        return softmax(self.actions(h), dim=0), tanh(self.value(h))

    def play(self, program, steps=10):
        logging.debug("Playing {}".format(program))
        state = program.reset()
        for s in range(steps):
            pi = mcts(
                    self, state, deepcopy(program), 
                    simulations=SIMS if self.train else SIMS // 2)
            pi = torch.tensor([n.pr() for n in pi])
            
            if self.training:
                # Policy Loss
                policy, v = self.forward(state)
                loss_policy = self.loss_policy(policy, pi.to(self.device))

                # Value Loss
                state, rew, done, info = program.step(torch.argmax(pi).item())
                loss_value = self.loss_value(v, torch.tensor(rew).to(self.device))

                # Optimizer Step
                loss = loss_policy + loss_value
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # MLFlow
                logging.debug(f"Step {s}, Action {max(pi).action} got reward {rew}")
                mlflow.log_metric("Loss", (loss_policy + loss_value).item())
                mlflow.log_metric(str(program), rew)
                mlflow.log_metric("Training Value Error", loss_value.item())
                mlflow.log_metric("Training Policy Error", loss_policy.item())
            else:
                state, rew, done, info = program.step(torch.argmax(pi).item())

                # MLFlow
                mlflow.log_metric("Testing {}".format(str(program)), rew)

            if done:
                break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    programs = Programs()
    train, test = programs[:65], programs[65:]
    model = AlphaCompile(train[0].observation_space, len(FLAGS) + 1)
    logging.info(model)

    model.train()
    for e in range(1, EPOCHS + 1):
        shuffle(train)
        for program in train:
            try:
                model.play(program)
            except Exception as e:
                logging.exception("{}: {}".format(repr(program), e), exc_info=True)
                continue
            except KeyboardInterrupt:
                save(model.state_dict(), join('source', 'AlphaCompile.pth'))
                break
            else:
                logging.info("Saving the model")
                save(model.state_dict(), join('source', 'AlphaCompile.pth'))

    model.eval()
    for program in test:
        try:
            model.play(program)
        except Exception as e:
            logging.exception("{}: {}".format(repr(program), e), exc_info=True)
            continue
        else:
            jit.save(jit.trace(model, (randn(program.observation_space),)), join('source', 'AlphaCompile.pt'))

