""" Main
Constructs graph, trains model, evaluates.

"""

from pickle import dump, load

from agent import Agent
from programs import Programs
from settings import *

AGENT_PATH = './save/agents/agent_{}_{}.pickle'
EPISODES = 100

metrics = logging.getLogger("metrics")
events = logging.getLogger("events")


def main():
    # Load Programs.
    programs = Programs()
    action_size = programs.num_actions

    for feature_set in Features:
        events.info(feature_set.name)

        # State size depends on the feature set in use.
        state_size = len(programs.programs[0].features[feature_set])

        for p in programs.programs_names:
            events.info(f"Withholding {p}")

            # Split the programs, load or build an agent.
            programs.split_programs(p)

            agent_path = AGENT_PATH.format(feature_set.name, p)
            try:
                with open(agent_path, 'rb') as a:
                    agent = load(a)
            except FileNotFoundError:
                agent = Agent(state_size, action_size, name=f"{feature_set.name}_{p}")

            agent.epsilon = 1

            # Each agent gets to try EPISODES times.
            for e in range(EPISODES + 1):
                program = programs.get_program()
                context = program.get_context(feature_set)
                actions = agent.act(context)

                try:
                    runtime = program.run(actions)
                    speedup = program.baseline / runtime

                    agent.remember(context, actions, runtime)
                    agent.log_stats("train_speedup", value=speedup)
                except OSError:
                    events.exception("Programs failed, please check reports.")

                if not e % 10:
                    events.info(f"Agent step {agent.step:>6d}: speedup {speedup:>4f}")

            # Save the agent.
            with open(agent_path, 'wb') as a:
                dump(agent, a)

            events.info(f"Testing against {p}")
            for program in programs.test:
                # Gather context for each program in testing.
                context = program.get_context(feature_set)

                # Set agent exploration to 0, and predict flags.
                agent.epsilon = 0
                actions = agent.act(context, num_return=5)

                # Compute baseline and optimized times.
                baseline = program.run([0])

                one_runtime = program.run(actions[:1])
                five_runtime = program.run(actions)

                one_speedup = baseline / one_runtime
                five_speedup = baseline / five_runtime

                # Log results.
                metrics.info(
                    f"{feature_set.name}, {program.prog_name}, {program.dataset}, {one_speedup}, {five_speedup}, {agent.step}")
                agent.log_stats("test_1_speedup", one_speedup)
                agent.log_stats("test_5_speedup", five_speedup)
                events.info(
                    f"Agent: {agent.name}, "
                    f"baseline: {baseline:>4f}, optimized: {five_runtime:>4f}, "
                    f"diff: {five_speedup:>4f}, "
                    f"flags used: {' '.join(ACTIONS[actions[0]])}"
                )


if __name__ == "__main__":
    main()
