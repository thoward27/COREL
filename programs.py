""" Bandit
Represents the problem being solved by the agent.

TODO:
TODO: Better error reporting when a program crashes
TODO: Generate static features
TODO: Generate dynamic features
TODO: Annotate types
TODO: Complete documentation
"""
import heapq
import pickle
import random
import re
import shlex
import subprocess
from decimal import Decimal
from threading import Thread

from settings import *

# Seed our random generator
random.seed()


class Program:
    """ A single executable program.
    """

    def __init__(self, benchmark: str, name: str, dataset: str, path: str, run: str, compile: str):
        self.benchmark = benchmark
        self.name = name
        self.dataset = dataset
        self._path = path
        self._str_run = run
        self._str_compile = compile

        self.full_name = "{}_{}_{}".format(self.benchmark, self.name, self.dataset)

        self.runtimes = [Decimal(0) for _ in range(len(ACTIONS))]
        self.features = {
            Features.STATIC: None,
            Features.DYNAMIC: None,
            Features.HYBRID: None
        }
        return

    def __repr__(self):
        return self.full_name

    def __str__(self):
        return "{}_{}".format(self.benchmark, self.name)

    def __eq__(self, other):
        return str(self) == str(other)

    def context(self, feature_set: Features) -> np.array:
        """ Returns the context of the feature_set. """
        if self.features[feature_set] is None:
            raise AttributeError("Features are not present.")

        # The pyTypeChecker merely flags what I've already caught above.
        # noinspection PyTypeChecker
        return np.reshape(self.features[feature_set], [1, -1])

    def valid(self) -> bool:
        """ Boolean validation. """
        return all([
            self.features[Features.STATIC] is not None,
            self.features[Features.DYNAMIC] is not None,
            self.features[Features.HYBRID] is not None
        ])

    def run(self, actions: list) -> Decimal:
        """ Runs the program.

        :param actions: A list of actions to try. Actions are integer indexes corresponding to the ACTIONS constant.
        """
        runtimes = []
        for action in actions:
            self._compile(' '.join(ACTIONS[action]))
            heapq.heappush(runtimes, self._run())
        return heapq.heappop(runtimes)

    def _compile(self, flags: str) -> None:
        result = subprocess.run(
            shlex.split(self._str_compile.format(flags)),
            shell=False,
            cwd=self._path,
            stdout=subprocess.PIPE
        )
        if not result.returncode == 0:
            events.error("Failed to compile " + self.full_name)
            raise OSError("Failed to compile")
        return

    def _run(self) -> Decimal:
        result = subprocess.run(
            shlex.split(self._str_run),
            shell=False,
            cwd=self._path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if not result.returncode == 0:
            events.error("Failed to run " + self.full_name)

        result = result.stderr.decode('utf-8')
        m = re.search(
            r'\nreal\t(?P<real>\d+m\d+.\d+)s\nuser\t(?P<user>\d+m\d+.\d+)s\nsys.(?P<sys>\d+m\d.\d+)s',
            result
        )

        # real_time = Decimal(self._compute_time(m.group('real')))
        user_time = self._compute_time(m.group('user'))
        syst_time = self._compute_time(m.group('sys'))

        return user_time + syst_time + Decimal("0.0001")

    def build_runtimes(self) -> None:
        self.runtimes = [self.run([i]) for i in range(len(ACTIONS))]
        return

    @staticmethod
    def _compute_time(group) -> Decimal:
        time = group.split('m')
        time = Decimal(time[0]) * 60 + Decimal(time[1])
        return time

    @property
    def baseline(self) -> Decimal:
        """ Returns a safe baseline. """
        if not self.runtimes[0]:
            self.runtimes[0] = self.run([0])
        return self.runtimes[0]

    @property
    def optimal(self) -> Decimal:
        """ Return the optimal. """
        return min(self.runtimes)


class Programs:
    """ A container of programs. """

    def __init__(self):
        self.programs = []

        from cbench import programs
        self.programs.extend(programs())

        self.programs = [p for p in self.programs if p.valid()]
        self.datasets = {p.dataset for p in self.programs}
        self.names = {str(p) for p in self.programs}

        self.save()

    def __iter__(self):
        for program in self.programs:
            yield program

    def save(self) -> None:
        with open('./save/programs.pickle', 'wb') as f:
            pickle.dump(self.programs, f, pickle.HIGHEST_PROTOCOL)
        return

    def build_runtimes(self) -> None:
        events.info("Building Runtimes.")
        self._build_runtimes_serial()
        return

    def filter(self, program_name: str) -> dict:
        """ Filters and returns a dictionary of 'training' and 'testing' programs. """
        ret = {
            'training': [],
            'testing': []
        }
        [(ret['testing'] if p == program_name else ret['training']).append(p) for p in self.programs]
        return ret

    def _build_runtimes_threaded(self):
        for count, dset in enumerate(self.datasets):
            events.info("Getting runtimes for dataset: %s/%s" % (count+1, len(self.datasets)))
            threads = [Thread(target=p.build_runtimes, name=p.full_name) for p in self.programs if p.dataset == dset]
            [thread.start() for thread in threads]
            [thread.join() for thread in threads]

    def _build_runtimes_serial(self):
        for program in self.programs:
            program.build_runtimes()
