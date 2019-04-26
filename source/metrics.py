""" Metrics files.
"""

import numpy as np
from numpy import argsort

import source.config as c
import source.utils as u
from source.programs import Program


class Metric:
    def keys(self) -> tuple:
        return tuple(k for k in self.__dict__.keys())

    def values(self) -> tuple:
        return tuple(v for v in self.__dict__.values())


class ProgramMetric(Metric):
    def __init__(self, feature_set: c.Features, program: Program, epoch: int, **kwargs):
        self.feature_set = str(feature_set).split('.')[-1]
        self.name = str(program.name)
        self.dataset = str(program.dataset)
        self.epoch = str(epoch)
        self.o3 = program.o3
        self.opt = program.optimal_runtime
        self.opt_actions = str(argsort(program.runtimes)[:10])
        self.__dict__.update(**kwargs)

    def print(self) -> None:
        print("; ".join([
            "E: {:0>5}".format(self.epoch),
            "{:20}".format(self.name + self.dataset),
            "wrt-03-1 {:5.3f}".format(self.o3 / getattr(self, "one_shot", 1)),
            "wrt-03-5 {:5.3f}".format(self.o3 / getattr(self, "five_shot", 1)),
            "wrt-03-10 {:5.3f}".format(self.o3 / getattr(self, "ten_shot", 1)),
            "action {:3d}".format(u.flags_to_action(getattr(self, "flags", [0, 0, 0, 0, 0, 0, 0]))),
            "opt-actions {}".format(self.opt_actions)
        ]))
        return


class ModelMetric(Metric):
    def __init__(self, feature_set, name, epoch, metric, value):
        self.feature_set = str(feature_set).split('.')[-1]
        self.name = str(name)
        self.epoch = str(epoch)
        self.metric = str(metric)
        self.value = str(value)

    def values(self):
        return self.feature_set, self.name, self.epoch, self.metric, self.value


def sql_type(python_type):
    if python_type == 'str':
        return "TEXT"
    elif python_type == 'Decimal':
        return "DEC"
    elif python_type == 'int':
        return "INT"
    elif python_type == 'float':
        return 'REAL'


def build_table(results):
    conn = c.sql.connect(c.RUN_DIR + '/data.db', detect_types=c.sql.PARSE_DECLTYPES)
    conn.execute(
        'create table {} ({})'.format(
            results[0].__class__.__name__,
            ', '.join(["{} {}".format(k, sql_type(v.__class__.__name__)) for k, v in results[0].__dict__.items()]))
    )
    conn.commit()
    conn.close()


def log(results: list, recover=True) -> None:
    try:
        conn = c.sql.connect(c.RUN_DIR + '/data.db')
        conn.executemany(
            'insert into {} values ({})'.format(
                results[0].__class__.__name__,
                ', '.join(['?' for _ in results[0].keys()])),
            [(r.values()) for r in results]
        )
        conn.commit()
        conn.close()
    except c.sql.OperationalError:
        if recover:
            build_table(results)
            log(results, recover=False)
        else:
            raise
    return


def actual(y_true: np.ndarray, _) -> np.ndarray:
    return np.mean(y_true)


def predicted(_, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(y_pred)
