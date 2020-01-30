import gym
import math

import numpy as np

from ml.reinforcement import QAgent

"""
States: Box(4)
    Num     observation             Min     Max
    1       Cart Position           -4.8    4.8
    2       Cart Velocity           -Inf    Inf
    3       Pole Angle              -24     24
    4       Pole Velocity at Tip    -Inf    Inf


Actions: Discrete(2)
    Num     Action
    0       Push cart to the left
    1       Push cart to the rigth
"""

env = gym.make('CartPole-v0')


# observation_space = [env.observation_space.low, env.observation_space.high]
upper_bounds = [env.observation_space.high[0], 0.5,
                env.observation_space.high[2], math.radians(50)]
lower_bounds = [env.observation_space.low[0], -0.5,
                env.observation_space.low[2], -math.radians(50)]

observation_space = [lower_bounds, upper_bounds]

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
gammas = [0.8, 0.85, 0.9, 0.95, 0.99]
epsilons = [0.1, 0.2, 0.3, 0.4]

current_max = 0
params = []

agent = QAgent(actions=[0, 1],
               observation_space=observation_space,
               buckets=(2, 4, 12, 4),
               epsilon=0.3,
               gamma=1.0,
               alpha=0.8)

iterations = []
for t in range(100000):
    current_state = env.reset()
    done = False

    i = 0
    while not done:
        env.render()
        action = agent.get_action(current_state)
        observation, reward, done, info = env.step(action)
        agent.update_q_table(
            current_state, action, reward, observation)
        current_state = observation
        i += 1

        # print('Game iterations', i)
        # print('Step', t)
        iterations.append(i)

print(np.mean(iterations))

env.close()
