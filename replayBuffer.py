import numpy as np

from constants import *

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ReplayBuffer():
    '''
    Replay buffer containing a fixed maximun number of trajectories with 
    the highest returns seen so far
    
    Params:
        size (int)
    
    Attrs:
        size (int)
        buffer (List of episodes)
    '''
    
    def __init__(self, size=0):
        self.size = size
        self.buffer = []
        
    def add(self, episode):
        '''
        Params:
            episode (namedtuple):
                (states, actions, rewards, init_command, total_return, length)
        '''
        
        self.buffer.append(episode)
    
    def get(self, num):
        '''
        Params:
            num (int):
                get the last `num` episodes from the buffer
        '''
        
        return self.buffer[-num:]
    
    def random_batch(self, batch_size):
        '''
        Params:
            batch_size (int)
        
        Returns:
            Random batch of episodes from the buffer
        '''
        
        idxs = np.random.randint(0, len(self), batch_size)
        return [self.buffer[idx] for idx in idxs]
    
    def sort(self):
        '''Keep the buffer sorted in ascending order by total return'''
        
        key_sort = lambda episode: episode.total_return
        self.buffer = sorted(self.buffer, key=key_sort)[-self.size:]
    
    def save(self, filename):
        '''Save the buffer in numpy format
        
        Param:
            filename (str)
        '''
        
        np.save(filename, self.buffer)
    
    def load(self, filename):
        '''Load a numpy format file
        
        Params:
            filename (str)
        '''
        
        raw_buffer = np.load(filename, allow_pickle=True)
        self.size = len(raw_buffer)
        self.buffer = \
            [make_episode(episode[0], episode[1], episode[2], episode[3], episode[4], episode[5]) \
             for episode in raw_buffer]
    
    def __len__(self):
        '''
        Returns:
            Size of the buffer
        '''
        return len(self.buffer)

if __name__ == "__main__":
    buf = ReplayBuffer()
    buf.load("buffer.npy")