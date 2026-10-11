states = ["left", "middle", "goal"]
actions = ["left", "right"]

def step(state, action):
    if state == "left" and action == "right":
        return "middle" , 0
    
    elif state == "middle" and action == "right":
        return "goal" , 1
    
    elif state == "middle" and action == "left":
        return "left" , 0
    
    elif state == "left" and action == "left":
        return "left" , 0

    elif state == "goal":
        return "goal", 0

    else: 
        # Function receives args with correct data type but inappropriate
        raise ValueError("Invalid State or action")


next_state, reward = step("middle", "right")
print(f"Next State : {next_state} and Reward: {reward}")