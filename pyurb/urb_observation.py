from settings.load import *
from pyurb.urb_coords import *
from pyurb.urb_imageio import *
from pyurb.urb_mappoint import *
import sys

class Observation:
    def __init__(self, frame, x, y):
        self.mappoint = None
        self.cx = x
        self.cy = y
        self.frame = frame
        self.leftx = int(x) - HALF_PATCH_SIZE
        self.topy =int(y) - HALF_PATCH_SIZE  
        self.keypoint = cv2.KeyPoint(x, y, 1, 0)
        self.z = None
        self.disparity = None
    
    def get_frame(self):
        return self.frame
    
    def set_mappoint(self, mappoint):
        self.mappoint = mappoint

    def register_mappoint(self):
        if self.mappoint is not None:
            self.mappoint.add_observation(self)
            
    def get_mappoint(self):
        return self.mappoint
    
    def get_mappoint_id(self):
        try:
            return self.mappoint.id
        except:
            return None
    
    def has_mappoint(self):
        return self.mappoint is not None

    def create_mappoint(self, id):
        self.mappoint = MapPoint(self)
        self.mappoint.id = id
    
    def get_patch(self):
        try:
            return self.patch
        except:
            self.patch = get_patch(self.get_frame().get_smoothed(), self.leftx, self.topy)
            self.latestpatch = self.patch
            return self.patch
        
    def get_patch_distance(self, keypoint):
        return cv2.norm(self.get_patch(), keypoint.get_patch(), NORM)
    
    def get_disparity(self, frameRight):
        if self.disparity is None:
            self.confidence, self.disparity = patch_disparity(self, frameRight)
        return self.disparity
                                                    
    def get_depth(self):
        if self.z is None and self.disparity is not None:
            self.z = estimated_distance(self.disparity)
        return self.z
    
    def get_affine_coords(self):
        try:
            return self.affine_coords
        except:
            self.affine_coords = cam_to_affine_coords(self.cx, self.cy, self.get_depth())
            return self.affine_coords
    
    def get_keypoint(self):
        return self.keypoint
        
# OpenCV reverses coordinates, so the observation on top of an edge has a smaller y coordinate than the bottom of the same vertical edge
class ObservationTopRight(Observation):
    def __init__(self, frame, x, y):
        Observation.__init__(self, frame, x, y)
        self.topy = self.cy - 1
        self.leftx = self.cx - 1

class ObservationTopLeft(Observation):
    def __init__(self, frame, x, y):
        Observation.__init__(self, frame, x, y)
        self.topy = self.cy - 1
        self.leftx = self.cx - PATCH_SIZE + 1
        
class ObservationBottomLeft(Observation):
    def __init__(self, frame, x, y):
        Observation.__init__(self, frame, x, y)
        self.topy = self.cy - PATCH_SIZE + 1
        self.leftx = self.cx - PATCH_SIZE + 1
  
class ObservationBottomRight(Observation):
    def __init__(self, frame, x, y):
        Observation.__init__(self, frame, x, y)
        self.topy = self.cy - PATCH_SIZE + 1
        self.leftx = self.cx - 1
