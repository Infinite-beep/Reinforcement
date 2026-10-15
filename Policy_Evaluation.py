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

# -----------------------------------
# Defining the Environment Dynamics
# -----------------------------------

# The environment dynamics are defined by the transition probabilities and rewards for each state-action pair. 
# In this case, we will define a simple deterministic environment where each action leads to a specific next state with a certain reward.

# Actions: 0: Up, 1: Right, 2: Down, 3: Left
actions = [0, 1, 2, 3]
action_probability = 0.25  # Uniform random policy pi(a|s) = 1/4 for all actions

def get_next_state_and_reward(state, action):
    # Define Environment dynamics: p(s', r | s, a)
    # Given the current state and action, return the next state and reward

    # If we are in a terminal state, we stay in the same state with reward 0
    if state in terminal_states:
        return state, 0
    
    #Convert 1D state to 2D coordinates
    row = state // 4
    col = state % 4

    # Execute action
    if action == 0:
        row = max(0, row - 1)  # Up
    elif action == 1:
        col = min(3, col + 1)  # Right
    elif action == 2:
        row = min(3, row + 1)  # Down   
    elif action == 3:
        col = max(0, col - 1)  # Left

    # Convert 2D coordinates back to 1D state
    next_state = row * 4 + col
    reward = -1.0  # Default reward for each step. 
    # why is reward -1: because we want to encourage the agent to reach the terminal states as quickly as possible, 
    # so we penalize each step taken.
    

    return next_state, reward

iteration = 0
while True:
    delta = 0 # Initialize max change for this iteration

    for s in range(number_states):
        if s in terminal_states:
            continue  # Skip terminal states

        #store the old value to measure the change
        v_old = V[s]
        expected_value = 0

        #sum over all actions under the uniform random policy
        for a in actions:
            next_state, reward = get_next_state_and_reward(s, a)
            #In-place update uses newly computed values immediately
            expected_value += action_probability * (reward + gamma * V[next_state])

        #Overwrite the value function with the new estimate
        V[s] = expected_value

        # Update the maximum change observed in this iteration
        delta = max(delta, abs(v_old - V[s]))\

    iteration += 1

    #Check  stopping criterion:halt when delta falls below theta
    if delta < theta:
        break

    #-- Display Results ---
    print(f"Policy evaluation converged after {iteration} iterations.\n")
    print("Final Estimated State-Value Function V(s):")
    print(np.reshape(V, (4, 4)), 1)

    


