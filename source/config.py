""" Settings file.
"""
import os
import sqlite3 as sql
from datetime import datetime
from decimal import Decimal
from enum import Enum

sql.register_adapter(Decimal, lambda d: str(d))
sql.register_converter("DEC", lambda s: Decimal(s.decode('utf-8')))


class Features(Enum):
    HYBRID = 0
    STATIC = 1
    DYNAMIC = 2


EPOCHS = 100
THREADED_RUNTIMES = False
USE_RUNTIMES = True
NOW = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
RUN_DIR = os.path.abspath('./runs/run_{}'.format(NOW))

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

ACTIONS = [(
    '-O3',
    *(f for f, s in zip(FLAGS, list(format(a, '0%sb' % N_FLAGS))) if s == '1')
) for a in range(2 ** N_FLAGS)]

N_ACTIONS = len(ACTIONS)

# noinspection SpellCheckingInspection
LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': '%(levelname)s: %(filename)s: %(message)s',
        },
        'detailed': {
            'class': 'logging.Formatter',
            'format': '%(asctime)s, %(levelname)-6s, %(filename)-6s, %(funcName)s, %(message)s',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple',
        },
        'event_file': {
            'class': 'logging.FileHandler',
            'level': 'INFO',
            'formatter': 'detailed',
            'filename': '{}/events.log'.format(RUN_DIR),
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', 'event_file'],
            'level': 'DEBUG',
            'propogate': False,
        }
    }
}
