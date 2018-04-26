""" Main
Constructs graph, trains model, evaluates.

"""
import random
from pickle import dump, load

from agent import Agent
from programs import Programs
from settings import *

EPISODES = 10


def main():
    # Load Programs.
    programs = Programs()
    action_size = len(ACTIONS)

    for feature_set in Features:
        events.info(feature_set.name)

        # State size depends on the feature set in use.
        state_size = len(programs.programs[0].features[feature_set])

        for p in programs.programs_names:
            # TODO: Multi-thread the following loop.

            events.info("Withholding: " + p)

            # Split the programs, load or build an agent.
            progs = programs.filter(p)

            agent_path = AGENT_PATH.format(feature_set.name, p)
            try:
                with open(agent_path, 'rb') as a:
                    agent = load(a)
            except FileNotFoundError:
                agent = Agent(state_size, action_size, name="{}_{}".format(feature_set.name, p))

            agent.epsilon = 1

            # Each agent gets to try EPISODES times.
            for e in range(EPISODES + 1):
                program = random.choice(progs['training'])
                context = program.context(feature_set)
                actions = agent.act(context)

                try:
                    runtime = program.run(actions)
                    speedup = program.baseline / runtime

                    agent.remember(context, actions, runtime)
                    agent.log_stats("train_speedup", value=speedup)
                except OSError:
                    events.exception("Programs failed, please check reports.")

                if not e % 10:
                    events.info("Agent step {:>6d}: speedup {:>4f}".format(agent.step, speedup))

            # Save the agent.
            with open(agent_path, 'wb') as a:
                dump(agent, a)

            events.info("Testing against " + p)
            for program in progs['testing']:
                # Gather context for each program in testing.
                context = program.context(feature_set)

                # Set agent exploration to 0, and predict flags.
                agent.epsilon = 0
                actions = agent.act(context, num_return=5)

                # Compute baseline and optimized times.
                baseline = program.run([0])
                optimal = min(program.runtimes)

                one_runtime = program.run(actions[:1])
                five_runtime = program.run(actions)

                one_speedup = baseline / one_runtime
                five_speedup = baseline / five_runtime
                optimal_speedup = (baseline / optimal) if optimal else 0

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


if __name__ == "__main__":
    main()
