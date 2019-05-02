from torch import nn

from Benchmarks import Programs
from PyTorchRL.agents.AlphaZero import AlphaZero
from source.config import FLAGS

STEPS = 10


class AlphaCompile(AlphaZero):
    def __init__(self, input_dim: int, output_dim: int = len(FLAGS) + 1):
        self.hidden = 5
        body = nn.Sequential(
            nn.Linear(input_dim, self.hidden),
            nn.Dropout(0.5),
            nn.Linear(self.hidden, self.hidden),
        )
        super().__init__(input_dim, output_dim, body)


if __name__ == "__main__":
    programs = Programs()
    train, test = programs.filter(programs[0])
    model = AlphaCompile(train[0].observation_space, len(FLAGS) + 1)
    for program in train:
        model.train()
        model.play(program)

        model.eval()
        model.play(program)

    for program in test:
        model.eval()
        model.play(program)
