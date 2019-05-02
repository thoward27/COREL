""" Basic configurations.
"""

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
