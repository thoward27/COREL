""" Main
Constructs graph, trains model, evaluates.

"""
import random
from pickle import dump, load

from agent import Agent
from programs import Programs
from settings import *

EPISODES = 100


def main():
    # Load Programs.
    programs = Programs()
    # programs.build_runtimes()
    # programs.save()

    for feature_set in Features:
        events.info(feature_set.name)

        for program in programs.names:
            events.info("Withholding: %s" % program)

            progs = programs.filter(program)

            # agent = load_agent(progs['testing'][0], feature_set)
            agent = Agent(progs['testing'][0], feature_set)

            train(agent, progs['training'], feature_set)

            save_agent(agent)

            test(agent, progs['testing'], feature_set)


def test(agent, programs, feature_set):
    events.info("Testing against %s" % programs[0])
    agent.epsilon = 1
    for program in programs:
        # Gather context for each program in testing.
        context = program.context(feature_set)

        # Set agent exploration to 0, and predict flags.
        agent.epsilon = 0
        actions = agent.act(context, num_return=5)

        # Compute baseline and optimized times.
        baseline = program.baseline
        optimal = program.optimal

        one_runtime = program.run(actions[:1])
        five_runtime = program.run(actions)

        try:
            one_speedup = baseline / one_runtime
            five_speedup = baseline / five_runtime
            optimal_speedup = baseline / optimal
        except ZeroDivisionError:
            events.exception("ZeroDivision during testing %s" % program.full_name, exc_info=True)
        else:
            # Report Results
            metrics.info(
                "{}, {}, {:>5f}, {:>5f}, {:>5f}, {:>5f}, {:>5f}, {:>5f}, {:>5f}, {}".format(
                    feature_set.name,
                    program.full_name,
                    baseline,
                    optimal,
                    one_runtime,
                    five_runtime,
                    one_speedup,
                    five_speedup,
                    optimal_speedup,
                    agent.step
                )
            )
            events.info(
                "Program: {}; Baseline: {:>4f}; Optimized: {:>4f}; Diff: {:>4f}; Flags Used: {}".format(
                    program.full_name,
                    baseline,
                    five_runtime,
                    five_speedup,
                    ' '.join(ACTIONS[actions[0]])
                )
            )


def save_agent(agent):
    """ Save the provided agent to path. """
    with open(agent.save_path, 'wb') as a:
        dump(agent, a)


def load_agent(program, feature_set):
    agent_path = AGENT_PATH.format("%s_%s" % (feature_set.name, program))
    try:
        with open(agent_path, 'rb') as a:
            agent = load(a)
    except FileNotFoundError:
        agent = Agent(program, feature_set)
    return agent


def train(agent, programs, feature_set):
    events.info("Training agent %s" % agent)
    agent.epsilon = 1
    for e in range(EPISODES + 1):
        program = random.choice(programs)
        context = program.context(feature_set)
        action = agent.act(context)

        try:
            runtime = program.run(action)
            speedup = program.baseline / runtime

        except OSError:
            events.exception("OSError: %s %s" % (program.full_name, action), exc_info=True)

        except ZeroDivisionError:
            events.exception(
                "Runtime was zero for %s, compiling with %s" % (program.full_name, ' '.join(ACTIONS[action[0]])),
                exc_info=True
            )

        else:
            agent.remember(context, action, runtime)
            agent.log_stats("train_speedup", value=speedup)

            if not e % 10:
                events.info("Agent step {:>6d}: speedup {:>4f}".format(agent.step, speedup))


if __name__ == "__main__":
    main()
