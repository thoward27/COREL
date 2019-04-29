from mlflow import log_param
from torch import nn

from PyTorchRL.agents.AlphaZero import AlphaZero
from source.config import FLAGS

STEPS = 10


class AlphaCompile(AlphaZero):
    def __init__(self, input_dim: int = 5, output_dim: int = len(FLAGS) + 1):
        hidden = 5
        body = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden),
        )
        super().__init__(input_dim, output_dim, body)


def train(programs):
    model = AlphaCompile()
    for program in programs:
        model.train()
        model.play(program)

        model.eval()
        model.play(program)


if __name__ == "__main__":
    log_param("steps", STEPS)
    train([])
