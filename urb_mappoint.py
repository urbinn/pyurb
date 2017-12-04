from settings.load import *
from urb_coords import *
from urb_imageio import *

class MapPoint:
    def __init__(self, obs):
        self.id = None
        self.affine_coords = obs.get_affine_coords()
        self.observations = [obs]
    
    def get_affine_coords(self):
        return self.affine_coords
    
    def add_observation(self, observation):
        self.observations.append(observation)
        
    def remove_observation(self, observation):
        self.observations.remove(observation)
        
    def get_observations(self):
        return self.observations
    