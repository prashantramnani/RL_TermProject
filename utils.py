import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from constants import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def generate_episode(env, policy, init_command=[1, 1]):
    '''
    Generate an episode using the Behaviour function.
    
    Params:
        env (OpenAI Gym Environment)
        policy (func)
        init_command (List of float) -- default [1, 1]
    
    Returns:
        Namedtuple (states, actions, rewards, init_command, total_return, length)
    '''
    
    command = init_command.copy()
    desired_return = command[0]
    desired_horizon = command[1]
    
    states = []
    actions = []
    rewards = []
    
    time_steps = 0
    done = False
    total_rewards = 0
    state = env.reset().tolist()
    
    while not done:
        # env.render()
        state_input = torch.FloatTensor(state).to(device)
        command_input = torch.FloatTensor(command).to(device)
        action = policy(state_input, command_input)
        next_state, reward, done, _ = env.step(action)
        
        # Modifying a bit the reward function punishing the agent, -100, 
        # if it reaches hyperparam max_steps. The reason I'm doing this 
        # is because I noticed that the agent tends to gather points by 
        # landing the spaceshipt and getting out and back in the landing 
        # area over and over again, never switching off the engines. 
        # The longer it does that the more reward it gathers. Later on in 
        # the training it realizes that it can get more points by turning 
        # off the engines, but takes more epochs to get to that conclusion.
        if not done and time_steps > max_steps:
            done = True
            reward = max_steps_reward
        
        # Sparse rewards. Cumulative reward is delayed until the end of each episode
#         total_rewards += reward
#         reward = total_rewards if done else 0.0
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state.tolist()
        
        # Clipped such that it's upper-bounded by the maximum return achievable in the env
        desired_return = min(desired_return - reward, max_reward)
        
        # Make sure it's always a valid horizon
        desired_horizon = max(desired_horizon - 1, 1)
    
        command = [desired_return, desired_horizon]
        time_steps += 1
    # env.close()    
    return make_episode(states, actions, rewards, init_command, sum(rewards), time_steps)

def sample_command(buffer, last_few):
    '''Sample a exploratory command
    
    Params:
        buffer (ReplayBuffer)
        last_few:
            how many episodes we're gonna look at to calculate 
            the desired return and horizon.
    
    Returns:
        List of float -- command
    '''
    if len(buffer) == 0: return [1, 1]
    
    # 1.
    commands = buffer.get(last_few)
    
    # 2.
    lengths = [command.length for command in commands]
    desired_horizon = round(np.mean(lengths))
    
    # 3.
    returns = [command.total_return for command in commands]
    mean_return, std_return = np.mean(returns), np.std(returns)
    desired_return = np.random.uniform(mean_return, mean_return+std_return)
    
    return [desired_return, desired_horizon]

