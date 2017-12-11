from pyurb.settings.load import *
from pyurb.urb_coords import *
from pyurb.urb_imageio import *

class MapPoint:
    def __init__(self, obs):
        self.id = None
        self.z = obs.get_depth()
        self.affine_coords = obs.get_affine_coords()
        self.observations = { obs }
    
    def get_affine_coords(self):
        return self.affine_coords
    
    def update_affine_coords(self, obs):
        if obs.get_depth() is not None and obs.z < self.z:
            self.z = obs.z
            self.affine_coords = obs.get_affine_coords()
    
    def add_observation(self, observation):
        self.observations.add(observation)
        
    def remove_observation(self, observation):
        self.observations.remove(observation)
        
    def get_observations(self):
        return self.observations
    