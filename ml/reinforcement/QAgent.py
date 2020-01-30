import numpy as np

EPSILON_MIN = 0.001
EPSILON_DECAY = 0.0001

ALPHA_MIN = 0.1
ALPHA_DECAY = 0.0001


class QAgent:
    def __init__(self, actions, observation_space, buckets, epsilon=0.2, gamma=0.9, alpha=0.5):
        self.buckets = buckets
        self.alpha = alpha  # Learning rate
        self.actions = actions
        self.epsilon = epsilon  # Exploration rate
        self.gamma = gamma  # Discount factor
        self.Q = np.zeros(self.buckets + (len(self.actions),))
        self.observation_space_min = observation_space[0]
        self.observation_space_max = observation_space[1]

    def get_q_value(self, state, action):
        return self.Q[state][action]

    def get_action(self, state):
        discretized_state = self.discretize(state)

        # Epsilon-greedy
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY

        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[discretized_state])
        else:
            return np.random.choice([a for a in range(len(self.actions))])

    def update_q_table(self, state, action, reward, next_state):
        if self.alpha > ALPHA_MIN:
            self.alpha -= ALPHA_DECAY

        discretized_state = self.discretize(state)
        discretized_next_state = self.discretize(next_state)

        self.Q[discretized_state][action] += self.alpha * (reward + self.gamma * np.max(
            self.Q[discretized_next_state]) - self.Q[discretized_state][action])

    def discretize(self, obs):
        ratios = [(obs[i] + abs(self.observation_space_min[i])) /
                  (self.observation_space_max[i] - self.observation_space_min[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i]))
                   for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i]))
                   for i in range(len(obs))]
        return tuple(new_obs)
