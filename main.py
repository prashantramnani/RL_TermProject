import gym

from utils import *
from replayBuffer import *
from constants import *

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

env = gym.make('LunarLander-v2') # RocketLander-v0 | LunarLander-v2 | MountainCar-v0 | CartPole-v0
_ = env.seed(seed)

def initialize_replay_buffer(replay_size, n_episodes, last_few):
    '''
    Initialize replay buffer with warm-up episodes using random actions.
    See section 2.3.1
    
    Params:
        replay_size (int)
        n_episodes (int)
        last_few (int)
    
    Returns:
        ReplayBuffer instance
    '''
    # This policy will generate random actions. Won't need state nor command
    random_policy = lambda state, command: np.random.randint(env.action_space.n)
    
    buffer = ReplayBuffer(replay_size)
    
    for i in range(n_episodes):
        command = sample_command(buffer, last_few)
        episode = generate_episode(env, random_policy, command) # See Algorithm 2
        buffer.add(episode)
    
    buffer.sort()
    return buffer

def initialize_behavior_function(state_size, 
                                 action_size, 
                                 hidden_size, 
                                 learning_rate, 
                                 command_scale):
    '''
    Initialize the behaviour function. See section 2.3.2
    
    Params:
        state_size (int)
        action_size (int)
        hidden_size (int) -- NOTE: not used at the moment
        learning_rate (float)
        command_scale (List of float)
    
    Returns:
        Behavior instance
    
    '''
    
    behavior = Behavior(state_size, 
                        action_size, 
                        hidden_size, 
                        command_scale)
    
    behavior.init_optimizer(lr=learning_rate)
    
    return behavior    

def UDRL(env, buffer=None, behavior=None, learning_history=[]):
    '''
    Upside-Down Reinforcement Learning main algrithm
    
    Params:
        env (OpenAI Gym Environment)
        buffer (ReplayBuffer):
            if not passed in, new buffer is created
        behavior (Behavior):
            if not passed in, new behavior is created
        learning_history (List of dict) -- default []
    '''

    if buffer is None:
        buffer = initialize_replay_buffer(replay_size, 
                                          n_warm_up_episodes, 
                                          last_few)

    if behavior is None:
        behavior = initialize_behavior_function(state_size, 
                                                action_size, 
                                                hidden_size, 
                                                learning_rate, 
                                                [return_scale, horizon_scale])                                          


if __name__ == "__main__":
    buffer = initialize_replay_buffer(replay_size, n_warm_up_episodes, last_few)
    print(len(buffer))