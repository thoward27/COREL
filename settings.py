""" Settings file. """

import logging.config
from enum import Enum
from itertools import chain, combinations

import numpy as np

LOAD_PROGRAMS = True
AGENT_PATH = './save/agents/agent_{}_{}.pickle'
C_BENCH = './cBench'


class Features(Enum):
    STATIC = 0
    DYNAMIC = 1
    HYBRID = 2


NUM_FLAGS = 7

FLAGS = [
    '-funsafe-math-optimizations',
    '-fno-guess-branch-probability',
    '-fno-ivopts',
    '-fno-tree-loop-optimize',
    '-fno-inline-functions',
    '-funroll-all-loops',
    '-O2'
]

ACTIONS = np.array(list(chain.from_iterable(combinations(FLAGS, n) for n in range(NUM_FLAGS + 1))))

LOG_CONFIG = {
    'version': 1,
    'formatters': {
        'detailed': {
            'class': 'logging.Formatter',
            'format': '%(asctime)s, %(levelname)-6s, %(filename)-6s, %(funcName)s, %(message)s'
        },
        'performance': {
            'class': 'logging.Formatter',
            'format': '%(asctime)s, %(message)s',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
        },
        'event_file': {
            'class': 'logging.FileHandler',
            'filename': './logs/events.log',
            'mode': 'w',
            'formatter': 'detailed',
            'level': 'WARNING'
        },
        'metric_file': {
            'class': 'logging.FileHandler',
            'filename': './logs/metrics.log',
            'mode': 'w',
            'formatter': 'performance',
        }
    },
    'loggers': {
        'metrics': {
            'handlers': ['metric_file'],
            'propagate': False
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console', 'event_file']
    }
}

logging.config.dictConfig(LOG_CONFIG)

events = logging.getLogger("events")
metrics = logging.getLogger("metrics")
