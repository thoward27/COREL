""" Monte Carlo Tree Search for program compilation.

TODO: Save tree in between search calls.
"""
import logging
from copy import deepcopy
from functools import total_ordering

from math import log, sqrt
from torch import nn, Tensor

from source.config import FLAGS

c_base = 0.2
c_init = 0.5


@total_ordering
class Node:
    def __init__(self, state: Tensor, action: int, prior: float, parent=None):
        self.state: Tensor = state
        self.action: int = action
        self.parent: Node = parent
        self.done: bool = False

        self.children: list = []
        self.n: int = 1
        self.w: float = 0
        self.p: float = prior

    def __repr__(self) -> str:
        return "({}, {})".format(self.action, round(self.w, 2))

    def __eq__(self, other) -> bool:
        return self.q() + self.u() == other.q() + other.u()

    def __gt__(self, other) -> bool:
        return self.q() + self.u() > other.q() + other.u()

    def pr(self) -> float:
        return self.q() + self.u()

    def q(self) -> float:
        try:
            return self.w / self.n
        except ZeroDivisionError:
            return 0

    def c(self) -> float:
        try:
            return (log((1 + self.parent.n + c_base) / c_base)) + c_init
        except AttributeError:
            return c_init

    def u(self) -> float:
        try:
            return (self.c() * self.p * sqrt(self.parent.n)) / self.n
        except AttributeError:
            return self.p / self.n


def mcts(model: nn.Module, state: Tensor, game, simulations: int):
    # Start from current root
    policy, value = model.forward(state)
    logging.debug("MCTS Initial Policy: {}".format([round(p.item(), 3) for p in policy]))

    tree = [Node(state, a, p.item()) for a, p in enumerate(policy)]

    # Traverse to leaf
    for s in range(1, simulations + 1):
        logging.debug("MCTS # {}, Tree: {}".format(s, [round(t.pr(), 3) for t in tree]))
        node = max(tree)
        flags = [node.action]
        while node.children:
            node = max(node.children)
            flags.append(node.action)

        # The agent has selected "pass"
        if flags[-1] == len(FLAGS):
            node.done = True

        if not node.done:
            simulation = deepcopy(game)
            # Expand leaf
            state, value, done, info = simulation.step(' '.join([FLAGS[f] for f in flags]))
            node.done = done
            logging.debug("Compiled with flags {}, got reward {}".format(flags, round(value, 2)))

            policy, _ = model.forward(state)
            node.children = [Node(state, a, p.item(), parent=node) for a, p in enumerate(policy)]
        else:
            value = node.w

        # Backward step
        while node:
            node.n += 1
            node.w += value
            node = node.parent

    # return the new policy
    return tree
