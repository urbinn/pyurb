from settings.load import *
from urb_coords import *
from urb_imageio import *
from urb_mappoint import *

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
    
    def get_patch_distanceM(self, keypoint, x, y):
        patch = get_patch(keypoint.get_frame().get_smoothed(), keypoint.leftx + x, keypoint.topy + y)
        return cv2.norm(self.get_patch(), patch, NORM)
    
    def get_mono_patch_distance_m(self, keypoint, x, y):
        patch = get_patch(keypoint.get_frame().get_smoothed(), keypoint.leftx + x  + HALF_PATCH_SIZE - MONO_HALF_PATCH_SIZE, keypoint.topy + y, MONO_PATCH_SIZE)
        return cv2.norm(self.get_mono_patch(), patch, NORM)
    
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
        
class ObservationTop(Observation):
    def __init__(self, frame, x, y):
        Observation.__init__(self, frame, x, y)
        self.topy = self.cy

class ObservationBottom(Observation):
    def __init__(self, frame, x, y):
        Observation.__init__(self, frame, x, y)
        self.topy = self.cy - PATCH_SIZE
 