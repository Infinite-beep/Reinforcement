from Tiny_RL_agent import step
policy = {
    "left" : "right",
    "middle" : "right",
    "goal" : None
}


#Example of using the policy to take a step in the environment
current_state = "middle"
total_reward = 0

while current_state != "goal":
    # A policy tells the agent which action to take in a particular state.
    action_to_take = policy[current_state]

    next_state, reward = step(current_state, action_to_take)
    print(f"Current State: {current_state}, Action Taken: {action_to_take}, Next State: {next_state}, Reward: {reward}")

    #Cumulative reward is the sum of all rewards received by the agent during its interaction with the environment.
    total_reward += reward
    current_state = next_state

    #break the loop if the next state is goal to avoid infinite loop
    if next_state == "goal":
        break

print(f"Total Reward: {total_reward}")