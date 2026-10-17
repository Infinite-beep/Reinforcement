import numpy as np
import RL_Config as  config
# -----------------------------------
# Defining the Environment Dynamics
# -----------------------------------

# The environment dynamics are defined by the transition probabilities and rewards for each state-action pair. 
# In this case, we will define a simple deterministic environment where each action leads to a specific next state with a certain reward.


def get_next_state_and_reward(state, action):
    # Define Environment dynamics: p(s', r | s, a)
    # Given the current state and action, return the next state and reward

    # If we are in a terminal state, we stay in the same state with reward 0
    if state in config.terminal_states:
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




# -----Policy Evalaution-------- 
iteration = 0
while True:
    delta = 0 # Initialize max change for this iteration

    for s in range(config.number_states):
        if s in config.terminal_states:
            continue  # Skip terminal states

        #store the old value to measure the change
        v_old = config.V[s]
        expected_value = 0

        #sum over all actions under the uniform random policy
        for a in config.actions:
            next_state, reward = get_next_state_and_reward(s, a)
            #In-place update uses newly computed values immediately
            expected_value += config.action_probability * (reward + config.gamma * config.V[next_state])

        #Overwrite the value function with the new estimate
        config.V[s] = expected_value

        # Update the maximum change observed in this iteration
        delta = max(delta, abs(v_old - config.V[s]))\

    iteration += 1

    #Check stopping criterion:halts when delta falls below theta
    if delta < config.theta:
        break

    #-- Display Results ---
    print(f"Policy evaluation converged after {iteration} iterations.\n")
    print("Final Estimated State-Value Function V(s):")
    print(np.reshape(config.V, (4, 4)), 1)

    


