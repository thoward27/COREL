""" Settings file.
"""
from enum import Enum


class Features(Enum):
    HYBRID = 0
    STATIC = 1
    DYNAMIC = 2


EPOCHS = 100

# noinspection SpellCheckingInspection
FLAGS = [
    '-funsafe-math-optimizations',
    '-fno-guess-branch-probability',
    '-fno-ivopts',
    '-fno-tree-loop-optimize',
    '-fno-inline-functions',
    '-funroll-all-loops',
    '-O2'
]
N_FLAGS = len(FLAGS)

