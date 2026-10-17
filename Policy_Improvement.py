import numpy as np
import RL_Config as cfg
from Policy_Evaluation_no_stoppage import get_next_state_and_reward


#1. -----------Policy Evaluation Algorithm-----------------
iteration = 0
while True:
    delta = 0
    for s in range(cfg.number_states):
        if s in cfg.terminal_states:
            continue 
        v_old = cfg.V[s]
        #Evaluate based on CURRENT policy
        a = cfg.policy[s]
        next_state, reward = get_next_state_and_reward(s, a)
        cfg.V[s] = reward + cfg.gamma * cfg.V[next_state]
        delta = max(delta, abs(v_old - cfg.V[s]))
        if delta < cfg.theta:
            break

        #2.  -----------Policy Improvement Algorithm-----------------

        # The policy improvement algorithm is used to improve the policy based on the current value function.
        # Flag to check if the policy is stable
        policy_stable = True 

        # Loop through all states to check if the policy can be improved
        for s in range(cfg.number_states):
            # Skip terminal states
            if s in cfg.terminal_states:
                continue  

            # Rember what the policy Currently does
            old_action = cfg.policy[s]

            # Find the best actions by looking ahead across all actions
            # Greedy action selection step : choose the action that maximizes the expected value

            action_values = []
            for a in cfg.actions:
                next_state, reward = get_next_state_and_reward(s, a)
                action_values.append(reward + cfg.gamma * cfg.V[next_state])

            # Update the policy to take the action that maximizes the expected value
            best_action = cfg.actions[np.argmax(action_values)]
            cfg.policy[s] = best_action

            # Check if the policy has changed for this state
            if old_action != best_action:
                policy_stable = False

    iteration += 1

    if policy_stable:
        break  # Exit the loop if the policy is stable

    print(f"Policy Iteration converged in {iteration} major iterations.\n")
    print("Optimal Policy Actions (0:Up, 1:Right, 2:Down, 3:Left):")
    print(cfg.policy.reshape((4, 4)))
            


            