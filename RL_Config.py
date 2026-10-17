import numpy as np

# Environment and Algorithm parameters
#---------------------------------------
# 4x4 grid world
number_states = 16

# Top-left and Bottom right corners are terminal
terminal_states = [0, 15]

#Discount factor
gamma = 1.0

#Accuaracy threshold for stopping
theta = 1e-4

'''
Initialize V(s) to 0 for all states
We should initialize V(s) arbitrarily so we choose 0
'''

V = np.zeros(number_states)
print(V)
policy = np.zeros(number_states, dtype=int)  # Initialize policy to always take action 0 (Up)

# Actions: 0: Up, 1: Right, 2: Down, 3: Left
actions = [0, 1, 2, 3]
action_probability = 0.25  # Uniform random policy pi(a|s) = 1/4 for all actions