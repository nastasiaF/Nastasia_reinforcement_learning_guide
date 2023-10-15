import numpy as np

# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.99
initial_epsilon = 1.0
epsilon_decay = 0.9975
min_epsilon = 0.05

# Initialize the Q-table
# The shape is based on the observation space and action space
q_table_shape = (28**max_length, 26)  # Adjust based on the state and action space
q_table = np.zeros(q_table_shape)

def epsilon_greedy_policy(state):
    """
    Returns an action based on epsilon-greedy policy.
    """
    if np.random.uniform(0, 1) < epsilon:
        # Exploration: choose a random action
        return np.random.choice(26)
    else:
        # Exploitation: choose the action with max Q-value for the current state
        # Mask forbidden actions using the second element of the observation
        masked_q_values = np.copy(q_table[state[0]])
        # Mask by setting very low Q-value for letter already tried
        masked_q_values[state[1]] = -np.inf
        # return the action with the max Q-value
        return np.argmax(masked_q_values)