from collections import namedtuple 
make_episode = namedtuple('Episode', 
                          field_names=['states', 
                                       'actions', 
                                       'rewards', 
                                       'init_command', 
                                       'total_return', 
                                       'length', 
                                       ])

replay_size = 500
n_warm_up_episodes = 10
last_few = 75
max_reward = 250