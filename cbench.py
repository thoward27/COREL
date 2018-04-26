""" cBench Utilities
"""
import os
from csv import reader
from subprocess import run, PIPE

from programs import Program
from settings import *

ROOT_DIR = C_BENCH
DATASETS = 5
DATASTART = 1
LOOPS = 75

_programs = []


def test():
    """ Test the benchmark suite. """
    pass


def programs():
    """ Collects and returns programs. """
    _remove_dirs()
    _create_dirs()

    _collect_paths()
    _collect_dynamic()
    _collect_static()

    return _programs


def _collect_paths():
    events.info("Gathering list of benchmarks.")
    for path, _, _ in os.walk(ROOT_DIR):
        if 'src_work' in path:
            for i in range(DATASTART, DATASETS + 1):
                _programs.append(
                    Program(
                        benchmark="cBench",
                        name=str(path.split('/')[2]),
                        dataset=str(i),
                        path=path,
                        run="./__run {} {}".format(i, LOOPS),  # 75 == Number of loops to do.
                        compile="export CCC_OPTS='-w {}'; ./__compile gcc",
                    )
                )
    return


def _remove_dirs() -> None:
    """ Removes working directories """
    events.info("Deleting working directories for benchmarks.")
    create_dirs = run('./all__delete_work_dirs', cwd=C_BENCH, stdout=PIPE)
    if not create_dirs.returncode == 0:
        events.error(create_dirs.stderr)
        raise OSError("Cannot create working dirs.")
    return


def _create_dirs() -> None:
    events.info("Creating working directories for benchmarks.")
    create_dirs = run('./all__create_work_dirs', cwd=C_BENCH, stdout=PIPE)
    if not create_dirs.returncode == 0:
        events.error(create_dirs.stderr)
        raise OSError("Cannot create working dirs.")
    return


def _compile_all() -> None:
    """ Attempts to compile all cBench programs. """
    compile_all = run(['./all_compile', 'gcc'], cwd=C_BENCH, stdout=PIPE)
    if not compile_all.returncode == 0:
        events.error(compile_all.stderr)
        raise OSError("Cannot compile programs.")
    return


def _generate_dynamic():
    """ Generates dynamic features.

    TODO: Generate dynamic features.
    """
    for program in programs:
        results = run(
            "{} -t {} -- ./__run 1".format(PIN, MICA),
            cwd=program['path'],
            shell=True,
            stdout=PIPE,
            stderr=PIPE)
        if not results.returncode == 0:
            events.error(results.stdout)
            events.error("Feature collection failed")

    events.info("running table generation.")
    run("sh tableGen.sh", cwd=C_BENCH, shell=True, stdout=PIPE)
    raise NotImplemented


def _collect_dynamic() -> None:
    """ Collects dynamic features

    TODO: Flip the loop
    """
    with open('./cbench_data/dynamic_features.csv') as csvfile:
        r = reader(csvfile)
        for row in r:
            for program in [p for p in _programs if p.name == row[0] and p.dataset == row[1]]:
                program.features[Features.DYNAMIC] = np.array(row[2:])
                program.features[Features.HYBRID] = np.array(row[2:])
    return


def _collect_static():
    """ Collects static features.

    Currently collects features from COBAYN data.

    TODO: Flip the loop.
    """
    with open('./cbench_data/static_features.csv') as csvfile:
        r = reader(csvfile)
        for row in r:
            for program in [p for p in _programs if p.name == row[0]]:
                program.features[Features.STATIC] = np.array(row[2:])
                np.append(program.features[Features.HYBRID], np.array(row[2:]))
    return
