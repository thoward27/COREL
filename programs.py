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
import os
import random
import re
import subprocess
import sys
from csv import reader
from pickle import dump, load
from subprocess import run, PIPE

from settings import *

# Seed our random generator
random.seed()

# Class-Wide Constants
PIN = '/home/tom/Documents/pin34/pin'
MICA = '/home/tom/Documents/pin34/source/tools/MICA-master/obj-intel64/mica.so'
C_BENCH = './cBench'
SAVE_PROGRAMS = './save/programs/'

events = logging.getLogger("events")
metrics = logging.getLogger("metrics")


class Program:
    """ Abstract program framework. """

    def __init__(self, prog_name: str, dataset: int, full_name: str, path: str, str_run: str, str_compile: str):
        self.full_name = full_name
        self.dataset = dataset
        self.prog_name = prog_name
        self.path = path
        self.str_run = str_run
        self.str_compile = str_compile
        self.filename = f"{SAVE_PROGRAMS}/{self.full_name}.pickle"

        # Internal State.
        self.baseline = 0
        self.optimal = 0
        self.runtimes = []
        # One set of static, dynamic and hybrid are separated by dataset.
        self.features = {
            Features.STATIC: [],
            Features.DYNAMIC: [],
            Features.HYBRID: []
        }
        self._get_runtimes()

    def save(self):
        """ Save the program. """
        try:
            with open(self.filename, "wb") as f:
                dump(self.__dict__, f)
        except FileNotFoundError:
            os.mkdir(SAVE_PROGRAMS)
            self.save()
        return

    @classmethod
    def load(cls, f):
        return load(f)

    def to_array(self):
        """ Converts lists to arrays. """
        for key, feature_set in self.features.items():
            self.features[key] = np.array(feature_set, dtype=np.float64)

    def get_context(self, feature_set) -> np.array:
        """ Returns context. """
        return np.reshape(self.features[feature_set], [1, -1])

    def _compile(self, flags):
        """ Compiles the program. """
        result = subprocess.run(
            self.str_compile.format(' '.join(flags)),
            shell=True,
            cwd=self.path,
            stdout=subprocess.PIPE)
        if not result.returncode == 0:
            events.error(f"Failed to compile {self.full_name}")
            raise OSError("Failed to compile")

    def _run(self):
        result = subprocess.run(
            self.str_run,
            shell=True,
            cwd=self.path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if not result.returncode == 0:
            events.error(f"Failed to run {self.full_name}")
            raise OSError("Failed to run")

        result = result.stderr.decode('utf-8')
        m = re.search(
            r'\nreal\t(?P<real>\d+m\d+.\d+)s\nuser\t(?P<user>\d+m\d+.\d+)s\nsys.(?P<sys>\d+m\d.\d+)s',
            result
        )

        # real_time = self._compute_time(m.group('real'))
        user_time = self._compute_time(m.group('user'))
        syst_time = self._compute_time(m.group('sys'))

        return user_time + syst_time

    def run(self, actions: list) -> float:
        """ Runs the program, returning the runtime. """
        runtimes = []
        for action in actions:
            self._compile(ACTIONS[action])
            runtime = self._run()
            heapq.heappush(runtimes, runtime)
            self.runtimes[action] = runtime
        return heapq.heappop(runtimes)

    def _get_runtimes(self) -> None:
        actions = [i for i in range(len(ACTIONS))]
        for action in actions:
            sys.stdout.write(f"\rFinding optimal: {action}/{len(ACTIONS)}")
            runtime = self.run([action])
            # Save the information for later use.
            self.runtimes.append(runtime)
        sys.stdout.write(f"\r")
        events.info("Runtimes found.")

        self.optimized = min(self.runtimes)
        self.baseline = self.runtimes[0]
        return

    def __str__(self):
        return self.prog_name

    @staticmethod
    def _compute_time(group):
        time = group.split('m')
        time = float(time[0]) * 60 + float(time[1])
        return time

    def valid(self) -> bool:
        """ Returns whether or not a program is valid.
        """
        return all([
            (self.features[Features.STATIC]),
            (self.features[Features.STATIC]),
            (self.features[Features.HYBRID])
        ])


class Benchmark:
    """ Abstract Benchmark Class. """

    def __init__(self):
        self.programs = []
        self.ROOT_DIR = None
        return

    def get_programs(self, create_dirs=True) -> []:
        """ Returns an array of programs, annotated with static and dynamic features. """
        if create_dirs:
            self._remove_dirs()
            self._create_dirs()

        # Collect required paths.
        self._collect_paths()

        # Generate features
        self._collect_dynamic()
        self._collect_static()
        self._filter()
        return self.programs

    def _collect_paths(self):
        """ Collects paths to the programs. """
        pass

    def _collect_static(self):
        """ Gather static features. """
        pass

    def _generate_static(self):
        """ Generate static features. """
        pass

    def _collect_dynamic(self):
        """ Gather dynamic features. """
        pass

    def _generate_dynamic(self):
        """ Create dynamic features. """
        pass

    @staticmethod
    def _create_dirs():
        """ Create working directories. """
        pass

    @staticmethod
    def _remove_dirs():
        """ Remove working directories. """
        pass

    @staticmethod
    def _compile_all():
        """ Compiles all programs. """
        pass

    def _filter(self):
        """ Filters out invalid elements. """
        self.programs = [p for p in self.programs if p.valid()]
        return


class cBench(Benchmark):
    """ cBench Application Suite. """

    def __init__(self):
        super().__init__()
        self.ROOT_DIR = C_BENCH
        self.DATASETS = 5
        self.DATASTART = 1
        self.LOOPS = 75
        return

    def _collect_paths(self):
        events.info("Gathering list of benchmarks.")
        for path, _, _ in os.walk(self.ROOT_DIR):
            if 'src_work' in path:
                for i in range(self.DATASTART, self.DATASETS):
                    self.programs.append(
                        Program(
                            prog_name=path.split('/')[2],
                            dataset=i,
                            full_name=f"cBench_{path.split('/')[2]}_d{i}",
                            path=path,
                            str_run=f"./__run {i} {self.LOOPS}",  # 75 == Number of loops to do.
                            str_compile="export CCC_OPTS='-w {}'; ./__compile gcc",
                        )
                    )
        return

    @staticmethod
    def _remove_dirs() -> None:
        """ Removes working directories """
        events.info("Deleting working directories for benchmarks.")
        create_dirs = run('./all__delete_work_dirs', cwd=C_BENCH, stdout=PIPE)
        if not create_dirs.returncode == 0:
            events.error(create_dirs.stderr)
            raise OSError("Cannot create working dirs.")
        return

    @staticmethod
    def _create_dirs() -> None:
        events.info("Creating working directories for benchmarks.")
        create_dirs = run('./all__create_work_dirs', cwd=C_BENCH, stdout=PIPE)
        if not create_dirs.returncode == 0:
            events.error(create_dirs.stderr)
            raise OSError("Cannot create working dirs.")
        return

    @staticmethod
    def _compile_all() -> None:
        """ Attempts to compile all cBench programs. """
        compile_all = run(['./all_compile', 'gcc'], cwd=C_BENCH, stdout=PIPE)
        if not compile_all.returncode == 0:
            events.error(compile_all.stderr)
            raise OSError("Cannot compile programs.")
        return

    def _generate_dynamic(self):
        """ Generates dynamic features. """
        for program in self.programs:
            results = run(f"{PIN} -t {MICA} -- ./__run 1", cwd=program['path'], shell=True, stdout=PIPE, stderr=PIPE)
            if not results.returncode == 0:
                events.error(results.stdout)
                events.error("Feature collection failed")

        events.info("running table generation.")
        run("sh tableGen.sh", cwd=C_BENCH, shell=True, stdout=PIPE)
        raise NotImplemented

    def _collect_dynamic(self):
        """ Collects dynamic features """
        with open('./cbench_data/dynamic_features.csv') as csvfile:
            r = reader(csvfile)
            for row in r:
                for program in [p for p in self.programs if str(p) == row[0] and p.dataset == int(row[1])]:
                    program.features[Features.DYNAMIC] = row[2:]
                    program.features[Features.HYBRID] = row[2:]
        return

    def _collect_static(self):
        """ Collects static features.

        Currently collects features from COBAYN data.
        # TODO: Flip the loop.
        """
        with open('./cbench_data/static_features.csv') as csvfile:
            r = reader(csvfile)
            for row in r:
                for program in [p for p in self.programs if str(p) == row[0]]:
                    program.features[Features.STATIC] = row[2:]
                    program.features[Features.HYBRID].extend(row[2:])
        return


class Programs:
    """ Programs Interaction """

    def __init__(self, load_progs=False):
        """ Build program holder """
        # Set default state.
        self.programs_path = SAVE_PROGRAMS
        self.programs = []
        self.programs_names = set()
        self.test = []
        self.train = []

        # Load Programs, or build.
        if load_progs:
            self._load_programs()
        else:
            self._build_programs()
            self._save_programs()

        # Static helper variables:
        self.programs_names = {str(p) for p in self.programs}
        self.num_programs = len(self.programs)
        self.num_actions = len(ACTIONS)
        return

    def split_programs(self, test_name: str):
        """ Split training and testing data. """
        [(self.test if str(p) == test_name else self.train).append(p) for p in self.programs]
        return

    def get_program(self):
        """ Returns a random bandit. """
        program = random.choice(self.programs)
        return program

    def _build_programs(self):
        """ Build a list of programs. """
        # Benchmarks and other programs go here.
        try:
            self.programs.extend(cBench().get_programs())

            # Validation performed here.
            self.programs = [p for p in self.programs if p.valid]

            # Turn valid programs feature_sets into np arrays.
            for p in self.programs:
                p.to_array()
        except KeyboardInterrupt:
            self._save_programs()
            raise
        else:
            self.programs.sort(key=lambda p: p.full_name)
        return

    def _load_programs(self):
        """ Load programs via pickle object. """
        for program_path in os.listdir(self.programs_path):
            with open(program_path, "rb") as f:
                self.programs.append(Program.load(f))
        return

    def _save_programs(self):
        """ Save array of programs as pickle object. """
        for program in self.programs:
            program.save()
        return
