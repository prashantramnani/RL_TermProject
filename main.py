import gym

from utils import *
from replayBuffer import *
from constants import *
from Behavior import *

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)


env = gym.make("MountainCar-v0") # RocketLander-v0 | LunarLander-v2 | MountainCar-v0 | CartPole-v0 | Freeway-ram-v0
_ = env.seed(seed)
# print(env.observation_space)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n


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
        # print(episode.rewards)
        buffer.add(episode)
        # print(episode.rewards)
    
    buffer.sort()
    # quit()
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

def train_behavior(behavior, buffer, n_updates, batch_size):
    '''Training loop
    
    Params:
        behavior (Behavior)
        buffer (ReplayBuffer)
        n_updates (int):
            how many updates we're gonna perform
        batch_size (int):
            size of the bacth we're gonna use to train on
    
    Returns:
        float -- mean loss after all the updates
    '''
    all_loss = []
    for update in range(n_updates):
        episodes = buffer.random_batch(batch_size)

        batch_states = []
        batch_commands = []
        batch_actions = []

        for episode in episodes:
            # print("REWARD: ", episode.rewards)
            T = episode.length
            if(T <= 6):
                k = 2
            else:    
                k = np.random.randint(3, T - 3)
            x = 0
            for i in range(T - k + 1):
                # T = episode.length
                t1  = x#np.random.randint(0, T)
                t2  = x + k - 1#np.random.randint(t1+1, T+1)
                dh = t2 - t1
                dr = sum(episode.rewards[t1:t2])

                st1 = episode.states[t1]
                at1 = episode.actions[t1]

                batch_states.append(st1)
                batch_actions.append(at1)
                batch_commands.append([dr, dh])
                x += 1

        batch_states = torch.FloatTensor(batch_states).to(device)
        batch_commands = torch.FloatTensor(batch_commands).to(device)
        batch_actions = torch.LongTensor(batch_actions).to(device)

        pred = behavior(batch_states, batch_commands)
        loss = F.cross_entropy(pred, batch_actions)    
        behavior.optim.zero_grad()
        loss.backward()
        behavior.optim.step()
        
        all_loss.append(loss.item())
    
    return np.mean(all_loss)

def generate_episodes(env, behavior, buffer, n_episodes, last_few):
    '''
    1. Sample exploratory commands based on replay buffer
    2. Generate episodes using Algorithm 2 and add to replay buffer
    
    Params:
        env (OpenAI Gym Environment)
        behavior (Behavior)
        buffer (ReplayBuffer)
        n_episodes (int)
        last_few (int):
            how many episodes we use to calculate the desired return and horizon
    '''

    stochastic_policy = lambda state, command: behavior.action(state, command)

    for i in range(n_episodes_per_iter):
        command = sample_command(buffer, last_few)
        episode = generate_episode(env, stochastic_policy, command) # See Algorithm 2
        buffer.add(episode)
    
    # Let's keep this buffer sorted
    buffer.sort()

def evaluate_agent(env, behavior, command, render=False):
    '''
    Evaluate the agent performance by running an episode
    following Algorithm 2 steps
    
    Params:
        env (OpenAI Gym Environment)
        behavior (Behavior)
        command (List of float)
        render (bool) -- default False:
            will render the environment to visualize the agent performance
    '''
    behavior.eval()
    
    print('\nEvaluation.', end=' ')
        
    desired_return = command[0]
    desired_horizon = command[1]
    
    print('Desired return: {:.2f}, Desired horizon: {:.2f}.'.format(desired_return, desired_horizon), end=' ')
    
    all_rewards = []
    
    for e in range(n_evals):
        
        done = False
        total_reward = 0
        state = env.reset().tolist()
    
        while not done:
            if render: env.render()
            
            state_input = torch.FloatTensor(state).to(device)
            command_input = torch.FloatTensor(command).to(device)

            action = behavior.greedy_action(state_input, command_input)
            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            state = next_state.tolist()

            desired_return = min(desired_return - reward, max_reward)
            desired_horizon = max(desired_horizon - 1, 1)

            command = [desired_return, desired_horizon]
        
        if render: env.close()
        
        all_rewards.append(total_reward)
    
    mean_return = np.mean(all_rewards)
    print('Reward achieved: {:.2f}'.format(mean_return))
    file = open('LOG.txt', 'a') 
    f.write('Reward achieved: {:.2f}'.format(mean_return))
    f.close()
    
    behavior.train()
    
    return mean_return

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
    # quit()
    if behavior is None:
        behavior = initialize_behavior_function(state_size, 
                                                action_size, 
                                                hidden_size, 
                                                learning_rate, 
                                                [return_scale, horizon_scale])                                          
    f = open('LOG.txt', 'w')
    f.close()
    for i in range(1, n_main_iter + 1):
        
        mean_loss = train_behavior(behavior, buffer, n_updates_per_iter, batch_size)
        print('Iter: {}, Loss: {:.4f}'.format(i, mean_loss))
        f = open('LOG.txt', 'a') 
        f.write('Iter: {}, Loss: {:.4f} \n'.format(i, mean_loss))
        f.close()
        generate_episodes(env, 
                          behavior, 
                          buffer, 
                          n_episodes_per_iter,
                          last_few)

        if i % evaluate_every == 0:
            command = sample_command(buffer, last_few)
            mean_return = evaluate_agent(env, behavior, command)
            
            learning_history.append({
                'training_loss': mean_loss,
                'desired_return': command[0],
                'desired_horizon': command[1],
                'actual_return': mean_return,
            })
            
            if stop_on_solved and mean_return >= target_return: 
                break

            behavior.save("behavior_window.pth")
    # buffer.save("buffer.npy")
    return behavior, buffer, learning_history                  
    
def evaluate():
    behavior = Behavior(state_size, 
                        action_size, 
                        hidden_size, 
                        [return_scale, horizon_scale])
    
    behavior.init_optimizer(lr=learning_rate)

    behavior.load("behavior_freeway.pth")
    command = [100, 1]
    mean_return = evaluate_agent(env, behavior, command, render=True)
    print(mean_return)

if __name__ == "__main__":
    _, _, _ = UDRL(env)
    # evaluate()
    