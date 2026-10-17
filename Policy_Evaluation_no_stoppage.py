import numpy as np
import RL_Config as config

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



# Iterative Policy Evaluation Loop
# -----------------------------------
# Assuming a uniform random policy
action_probability = 0.25  

# Full sweep of all states
for s in range(config.number_states):
    if s in config.terminal_states:
        continue  # Skip terminal states

    expected_value = 0

    #Loop throught the actions
    for a in config.actions:
        # Get the next state and reward for the current state-action pair
        next_state, reward = get_next_state_and_reward(s, a)

        # Using the equation that tells us how to calculate the new value for a state using the old values of its neighbors
        # Probability of action * (reward + gamma * V(next_state))
        expected_value += config.action_probability * (reward + config.gamma * config.V[next_state])

    # Update the value function for the current state
    config.V[s] = expected_value
    print("Values after 1 sweep: ", np.round(config.V.reshape(4, 4), 1))