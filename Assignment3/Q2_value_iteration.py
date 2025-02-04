import numpy as np
import gym
import random
import math

env1 =["SHFF", "FFFH", "FHFH", "HFFG"]
env2= ["SFFFFF", "FFFHFF", "FHFHHH", "HFFFFG"]
env3 = ['SFFHFFHH', 'HFFFFFHF', 'HFFHHFHH', 'HFHHHFFF', 'HFHHFHFF', 'FFFFFFFH', 'FHHFHFHH', 'FHHFHFFG'] 

selectedEnv = env2
env = gym.make('FrozenLake-v1', desc=selectedEnv, render_mode="human", is_slippery = False)
env.reset()
env.render()

# change-able parameters:
discount_factor = 0.99
delta_threshold = 0.00001
epsilon = 1

def value_iteration(env, gamma=0.99, epsilon=1e-6):
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize the value function
    V = np.zeros(num_states)

    while True:
        delta = 0
        # Loop through each state
        for state in range(num_states):
            v = V[state]
            # Calculate the value for each action & find the max value
            V[state] = max([sum([prob * (reward + gamma * V[next_state])
                                for prob, next_state, reward, done in env.P[state][action]])
                            for action in range(num_actions)])
            
            # Calculate the maximum change in the value function for all states
            delta = max(delta, abs(v - V[state]))
        
        # Check for convergence
        if delta < epsilon:
            break

    # Extract the policy 
    policy = np.zeros(num_states, dtype=int)
    for state in range(num_states):
        # Find the best action for each state
        action_values = np.zeros(num_actions)
        for action in range(num_actions):
            for prob, next_state, reward, done in env.P[state][action]:
                action_values[action] += prob * (reward + gamma * V[next_state])
        best_action = np.argmax(action_values)
        policy[state] = best_action

    return policy, V

# Run value iteration
policy, V = value_iteration(env)

# Print results
print("Optimal Value Function:")
print(V.reshape(len(selectedEnv), len(selectedEnv[0])))

print("\nOptimal Policy (0=Left, 1=Down, 2=Right, 3=Up):")
print(policy.reshape(len(selectedEnv), len(selectedEnv[0])))

# resetting the environment and executing the policy
state = env.reset()
state = state[0]
step = 0
done = False
print(state)

max_steps = 100
for step in range(max_steps):

    # Getting max value against that state, so that we choose that action
    action = policy[state]
    new_state, reward, done, truncated,info = env.step(action) #information after taking the action

    env.render()
    if done:
        print("number of steps taken:", step)
        break

    state = new_state

env.close()