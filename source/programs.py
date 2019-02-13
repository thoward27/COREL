""" Bandit
Represents the problem being solved by the agent.
"""
import copy
import json
import logging
import os
import random
import re
import shlex
import subprocess
from builtins import OSError
from decimal import Decimal
from threading import Thread

import numpy as np

import source.config as config
from source.config import Features

# Seed our random generator
random.seed()

events = logging.getLogger(__name__)


class Program:
    """ A single executable program.

    Comparable, save-able, usable objects, that have been built for usability.
    """

    def __init__(self, benchmark: str, name: str, dataset: str, path: str, str_run: str, str_compile: str, **kwargs):
        """ Constructs a new program. All fields are required. "Benchmark + name + dataset" must be unique.

        :param benchmark: The benchmark the program belongs to (group ID)
        :param name: The name of the program
        :param dataset: The dataset to use. (Allows copies of a program to be instantiated)
        :param path: This is the CWD value to be used when executing the program
        :param str_run: A bash command to run the program. (daisy chaining is okay)
        :param str_compile: A bash command to compile the program (ie make ...), including {} for flag insertion.
        """
        self.benchmark = benchmark
        self.name = name
        self.dataset = dataset

        self.path = path
        self.str_run = str_run
        self.str_compile = str_compile

        self.runtimes = [Decimal("0") for _ in range(len(config.ACTIONS))]
        self.features = {
            Features.STATIC: np.array([]),
            Features.DYNAMIC: np.array([]),
            Features.HYBRID: np.array([])
        }

        self._baseline = None
        self._o2 = None
        self.__dict__.update(kwargs)
        return

    def __repr__(self) -> str:
        """ Unique representation of the program. """
        return "{}_{}_{}".format(self.benchmark, self.name, self.dataset)

    def __str__(self) -> str:
        """ A non-unique representation of the program. """
        return "{}_{}".format(self.benchmark, self.name)

    def __eq__(self, other) -> bool:
        """ Equality based on str() non-unique representation.

        Essentially this tests whether or not two programs are the same,
        expect for their dataset.
        """
        return str(self) == str(other)

    def __lt__(self, other) -> bool:
        """ This allows sorting by name, using the rep() unique representation. """
        return repr(self) < repr(other)

    def to_json(self) -> dict:
        """ Returns a JSON-Safe dictionary.

        This function allows easy saving and loading of programs.
        """
        d = copy.deepcopy(self.__dict__)
        d['runtimes'] = [str(x) for x in d['runtimes']]
        d['features'] = {str(key): value.tolist() for (key, value) in d['features'].items()}
        return d

    @staticmethod
    def from_json(obj: dict):
        """ Creates a new program based on the JSON Dict provided.

        :param obj: a JSON dictionary of the program.
        """
        obj['runtimes'] = [Decimal(x) for x in obj['runtimes']]
        obj['features'] = {eval(key): np.array(value) for (key, value) in obj['features'].items()}
        p = Program(**obj)
        return p

    def context(self, feature_set: config.Features) -> np.array:
        """ Returns the features corresponding to the given feature_set.

        :param feature_set: The feature set to use.
        """
        if self.features[feature_set] is None:
            raise AttributeError("Features are not present.")

        # The pyTypeChecker merely flags what I've already caught above.
        # noinspection PyTypeChecker
        return np.reshape(self.features[feature_set], [1, -1])

    def valid(self) -> bool:
        """ Returns whether or not a given program is valid, i.e. does it have all of it's features.  """
        return all([
            (self.features[config.Features.STATIC].size > 0),
            (self.features[config.Features.DYNAMIC].size > 0),
            (self.features[config.Features.HYBRID].size > 0)
        ])

    def run(self, action: int or str or tuple) -> Decimal:
        """ Runs the program with the given action.

        :param action: An action to try. Actions can be str or int.
        """
        # Parse the type of action we're running.
        if type(action) is int:
            idx = action
            action = ' '.join(config.ACTIONS[action])
        elif type(action) is np.int64:
            idx = int(action)
            action = ' '.join(config.ACTIONS[idx])
        elif type(action) is str:
            idx = None
            action = action
        elif type(action) is tuple:
            idx = config.ACTIONS.index(action)
            action = ' '.join(action)
        elif type(action) is np.ndarray:
            idx = int(''.join(["1" if i == 1.0 else "0" for i in np.round(action.flatten())]), 2)
            action = ' '.join(config.ACTIONS[idx])
        else:
            print(action)
            raise ValueError("Invalid action passed to run.")

        # Test whether or not to use existing runtimes, and if we can.
        if not (config.USE_RUNTIMES and idx is not None and self.runtimes[idx]):
            events.info("Running {} with {}".format(repr(self), action))
            try:
                self._compile(action)
            except OSError:
                events.exception("{} failed to compile with {}".format(repr(self), action), exc_info=True)
                raise
            finally:
                if idx is not None:
                    self.runtimes[idx] = self._run()
                else:
                    # This returns early if the action was a str, as there is no corresponding index.
                    return self._run()
        else:
            events.debug("Using cached time for {} with {}".format(repr(self), action))

        return self.runtimes[idx]

    def _compile(self, flags: str) -> None:
        result = subprocess.run(
            shlex.split(self.str_compile.format(flags)),
            shell=False,
            cwd=self.path,
            stdout=subprocess.PIPE
        )
        if not result.returncode == 0:
            events.error("Failed to compile {} with {}".format(repr(self), flags), exc_info=True)
            raise OSError("Failed to compile {}".format(repr(self)))
        return

    # noinspection SpellCheckingInspection
    def _run(self) -> Decimal:
        result = subprocess.run(
            shlex.split(self.str_run),
            shell=False,
            cwd=self.path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if not result.returncode == 0:
            events.error("Failed to run " + repr(self))

        result = result.stderr.decode('utf-8')
        m = re.search(
            r'\nreal\t(?P<real>\d+m\d+.\d+)s\nuser\t(?P<user>\d+m\d+.\d+)s\nsys.(?P<sys>\d+m\d.\d+)s',
            result
        )

        # real_time = self._compute_time(m.group('real'))
        user_time = self._compute_time(m.group('user'))
        syst_time = self._compute_time(m.group('sys'))

        return user_time + syst_time + Decimal("0.0001")

    def build_runtimes(self) -> None:
        """ Computes and saves all possible runtimes for a given program. """
        self.runtimes = [self.run(a) for a in config.ACTIONS]
        return

    @staticmethod
    def _compute_time(group) -> Decimal:
        time = group.split('m')
        time = Decimal(time[0]) * 60 + Decimal(time[1])
        return time

    @property
    def baseline(self) -> Decimal:
        """ Runtime of no flags. """
        if not config.USE_RUNTIMES:
            self._compile('')
            self._baseline = self._run()
        return self._baseline

    @property
    def o2(self) -> Decimal:
        """ Runtime of -O2. """
        if not config.USE_RUNTIMES:
            self._compile('-O2')
            self._o2 = self._run()
        return self._o2

    @property
    def o3(self) -> Decimal:
        """ Runtime of -O3. """
        if not config.USE_RUNTIMES:
            self._compile('-O3')
            self.runtimes[0] = self._run()
        return self.runtimes[0]

    @property
    def optimal_runtime(self) -> Decimal:
        """ The fastest runtime achievable.

        If BUILD_RUNTIMES is False, this may return 0.
        """
        return min(self.runtimes)

    @property
    def optimal_index(self) -> int:
        """ The corresponding index to the optimal runtime. """
        return self.runtimes.index(min(self.runtimes))


class Programs:
    """ An intelligent list of programs.

    Iterable, index-able, filterable, etc.
    """

    def __init__(self, programs=None):
        self.programs = programs if programs else self.benchmarks()

        self.programs = sorted([p for p in self.programs if p.valid()])
        self.datasets = {p.dataset for p in self.programs}
        self.names = sorted(list({str(p) for p in self.programs}))
        self.save()

    def __iter__(self):
        for program in self.programs:
            yield program

    def __getitem__(self, item):
        return self.programs[item]

    def __len__(self):
        return len(self.programs)

    def save(self) -> None:
        """ Saves the list of programs as a single JSON. """
        d = {'programs': [x.to_json() for x in self.programs]}
        try:
            with open(config.RUN_DIR + '/programs.json', 'w') as f:
                json.dump(d, f)
        except FileNotFoundError:
            os.makedirs(config.RUN_DIR)
            with open(config.RUN_DIR + '/programs.json', 'w') as f:
                json.dump(d, f)
        return

    @staticmethod
    def load(run_dir: str):
        """ Load cached programs.

        :param run_dir: The directory to find the 'programs.json' file.
        """
        with open(run_dir + "/programs.json", 'r') as f:
            programs = json.load(f)

        return Programs([Program.from_json(x) for x in programs['programs']])

    def build_runtimes(self) -> None:
        """ Collects all runtimes for every program in the list.

        Runtimes can be gathered in parallel, or serially, via changing c.THREADED_RUNTIMES.
        """
        self._build_runtimes_threaded() if config.THREADED_RUNTIMES else self._build_runtimes_serial()
        self.save()
        return

    def filter(self, program_name: str, validation_split: float = 0.2) -> dict:
        """ Filters and returns a dictionary of 'training' and 'testing' programs.

        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data.
        :param program_name: Represents the program to be tested against. All other programs are "training".
        """
        training = []
        testing = []
        [(testing if p == program_name else training).append(p) for p in self.programs]
        split_index = int(len(training) * validation_split)
        return {
            'training': training[split_index:],
            'validation': training[:split_index],
            'testing': testing
        }

    def _build_runtimes_threaded(self):
        # noinspection SpellCheckingInspection
        for count, dset in enumerate(self.datasets):
            threads = [Thread(target=p.build_runtimes, name=repr(p)) for p in self.programs if p.dataset == dset]
            [thread.start() for thread in threads]
            [thread.join() for thread in threads]

    def _build_runtimes_serial(self):
        [p.build_runtimes() for p in self.programs]

    @staticmethod
    def benchmarks():
        """ Collects all implemented Benchmark suites. """
        from benchmarks.cBench.cBench import cBench
        return list(cBench())
