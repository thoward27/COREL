import inspect
import json
import os
from datetime import datetime

import numpy as np

import source.config as c


def is_model(stackframe):
    return stackframe.function == 'main' and 'source/models' in stackframe.filename


def filename(path):
    return path.split('/')[-1].split('.')[0]


def build_directory():
    """ Build a valid run directory. """
    try:
        os.makedirs(c.RUN_DIR + '/models')
    except FileExistsError:
        pass

    with open(c.RUN_DIR + '/details.json', 'w') as f:
        json.dump({'details': {
            'EPOCHS': c.EPOCHS,
            'THREADED_RUNTIMES': c.THREADED_RUNTIMES,
            'USE_RUNTIMES': c.USE_RUNTIMES,
            'FLAGS': c.FLAGS,
            'ACTIONS': c.ACTIONS,
            'NOW': c.NOW,
            'LOG_CONFIG': c.LOG_CONFIG,
            'MODEL': filename([s.filename for s in inspect.stack() if is_model(s)][0])
        }}, f)
    return


def configure_loggers():
    """ Configure the loggers. """
    import logging.config
    log_c = c.LOG_CONFIG
    log_c['handlers']['event_file']['filename'] = c.RUN_DIR + '/events.log'
    logging.config.dictConfig(log_c)
    return


def init_run():
    """ Initializes a run. """
    build_directory()
    configure_loggers()
    return


def action_to_flags(action: int) -> np.array:
    """ Converts an integer action into a binary representation.
    Matches the neuron-output that would produce the given action.
    """
    if not 0 <= action <= 2 ** c.N_FLAGS:
        raise ValueError('Action must be within range attainable from flags.')
    return np.array([int(x) for x in list(format(action, '0' + str(c.N_FLAGS) + 'b'))])


def flags_to_action(flags: list or np.array) -> int:
    """ Converts neuron output to an action.
    Treats the flags as positions in a binary string, which is then
    converted to decimal in order to get an "action".
    """
    flags = list(flags.flatten()) if type(flags) is np.ndarray else flags

    if len(flags) != c.N_FLAGS:
        raise ValueError('Length of flags must equal the number of flags in use.')

    return int(''.join(["1" if i == 1.0 else "0" for i in np.round(flags)]), 2)


def run_path_to_datetime(path):
    path = path.split('/')[-1]  # remove prior directories
    path = path.split("_")[1:]  # exclude "run"
    return datetime(*[int(t) for t in path])
