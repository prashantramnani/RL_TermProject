from collections import namedtuple 
make_episode = namedtuple('Episode', 
                          field_names=['states', 
                                       'actions', 
                                       'rewards', 
                                       'init_command', 
                                       'total_return', 
                                       'length', 
                                       ])

replay_size = 1000
n_warm_up_episodes = 100
last_few = 150
max_reward = 350
hidden_size = 128
learning_rate = 0.0003
return_scale = 0.02
horizon_scale = 0.01
n_main_iter = 1200
n_updates_per_iter = 100
n_episodes_per_iter = 20
batch_size = 768
max_steps = 300
max_steps_reward = -50
evaluate_every = 10
n_evals = 1
stop_on_solved = False
target_return = 200