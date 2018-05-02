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
    programs.build_runtimes()
    programs.save()

    action_size = len(ACTIONS)

    for feature_set in Features:
        events.info(feature_set.name)

        # State size depends on the feature set in use.
        state_size = len(programs.programs[0].features[feature_set])

        for p_name in programs.programs_names:
            events.info("Withholding: %s" % p_name)

            # Setup the programs
            progs = programs.filter(p_name)

            # Setup the Agent

            agent = load_agent(action_size, feature_set, p_name, state_size)
            agent.epsilon = 1  # Introduce some randomness via epsilon manipulation.

            train_agent(agent, feature_set, progs)

            save_agent(agent)

            test_agent(agent, feature_set, progs)


def test_agent(agent, feature_set, progs):
    events.info("Testing against %s" % progs['testing'][0].name)
    for program in progs['testing']:
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


def load_agent(action_size, feature_set, p_name, state_size):
    agent_path = AGENT_PATH.format("%s_%s" % (feature_set.name, p_name))
    try:
        with open(agent_path, 'rb') as a:
            agent = load(a)
    except FileNotFoundError:
        agent = Agent(state_size, action_size, name="{}_{}".format(feature_set.name, p_name))
    return agent


def train_agent(agent, feature_set, progs):
    events.info("Training agent %s" % agent.name)
    for e in range(EPISODES + 1):
        program = random.choice(progs['training'])
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
