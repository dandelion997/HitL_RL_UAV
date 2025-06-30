# File func: config file
"""Configuration parameter file"""
class Config:
    def __init__(self):
        self.obs_dim = 9
        self.act_dim = 3
        self.actionBound = [[0.1,3],[0.1,3],[0.1,3]]

        self.MAX_EPISODE = 1000
        self.MAX_STEP = 500
        self.batch_size = 128

        self.update_every = 50
        self.noise = 0.3

        self.if_load_weights = False