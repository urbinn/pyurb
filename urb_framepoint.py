from urb_constants import *
import sys
from urb_coords import *
from urb_imageio import *

class FramePoint:
    def __init__(self, frame, x, y):
        self.id = None
        self.matches = None
        self.cx = x
        self.cy = y
        self.frame = frame
        self.leftx = int(x) - HALF_PATCH_SIZE
        self.topy =int(y) - HALF_PATCH_SIZE  
        self.keypoint = cv2.KeyPoint(x, y, 1, 0)
        self.z = None
        self.disparity = None
    
    def get_patch(self):
        try:
            return self.patch
        except:
            self.patch = get_patch(self.frame.get_smoothed(), self.leftx, self.topy)
            self.latestpatch = self.patch
            return self.patch
        
    def get_latest_patch(self):
        try:
            return self.latestpatch
        except:
            return self.get_patch()
        
    def get_mono_patch(self):
        try:
            return self.monopatch
        except:
            self.monopatch = get_patch(self.frame.get_smoothed(), self.leftx + HALF_PATCH_SIZE - MONO_HALF_PATCH_SIZE, self.topy, MONO_PATCH_SIZE)
            return self.monopatch
        
    def get_patch_distance(self, keypoint):
        return cv2.norm(self.get_patch(), keypoint.get_patch(), NORM)
    
    def get_mono_patch_distance(self, keyPoint):
        return cv2.norm(self.get_mono_patch(), keyPoint.get_mono_patch(), NORM)
    
    def get_patch_distanceM(self, keypoint, x, y):
        patch = get_patch(keypoint.frame.get_smoothed(), keypoint.leftx + x, keypoint.topy + y)
        return cv2.norm(self.get_patch(), patch, NORM)
    
    def get_mono_patch_distance_m(self, keypoint, x, y):
        patch = get_patch(keypoint.frame.get_smoothed(), keypoint.leftx + x  + HALF_PATCH_SIZE - MONO_HALF_PATCH_SIZE, keypoint.topy + y, MONO_PATCH_SIZE)
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
            return self.wcoords
        except:
            self.wcoords = cam_to_affine_coords(self.cx, self.cy, self.get_depth())
            return self.wcoords
    
    def get_keypoint(self):
        return self.keypoint
    
class FramePointTop(FramePoint):
    def __init__(self, frame, x, y):
        FramePoint.__init__(self, frame, x, y)
        self.topy = self.cy

class FramePointBottom(FramePoint):
    def __init__(self, frame, x, y):
        FramePoint.__init__(self, frame, x, y)
        self.topy = self.cy - PATCH_SIZE
        