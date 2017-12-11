from settings.load import *
from pyurb.urb_observation import *
from pyurb.urb_imageio import *
import cv2
import numpy as np
from pyurb.urb_filter import *
import pyurb.urbg2o as urbg2o

def observations_to_numpy(observations):
    #fps = [(fp.get_mappoint().id, fp.get_mappoint().get_affine_coords(), fp.cx, fp.cy) for fp in observations if fp.has_mappoint()]
    fps = [(fp.get_mappoint().get_affine_coords(), fp.cx, fp.cy) for fp in observations if fp.has_mappoint()]
    fps =[(1,m[0], m[1], m[2], x, y) for m, x, y in fps]
    arr = np.array(fps, dtype=np.float64, order='f')
    return arr

def get_pose(observations):
    pose = np.ndarray((4,4), dtype=np.float64, order='f')
    fps = observations_to_numpy(observations)
    pointsLeft = urbg2o.poseOptimization(fps, pose)
    #print(pointsLeft, pose)
    return pose, pointsLeft

class Frame:
    def __init__(self, filepath, rightpath = None):
        self.id = None
        self._filepath = filepath
        self._pose = None
        self._keyframe = None
        self._rightpath = rightpath
        
    def get_right_frame(self):
        try:
            return self._rightframe
        except:
            if self._rightpath is None:
                raise ValueError('rightpath is not set')
            self._rightframe = Frame('/'.join([self._rightpath, self._filepath.split('/')[-1]]))
            return self._rightframe
        
    def set_pose(self, pose):
        self._pose = pose
     
    def get_image(self):
        try:
            return self._image
        except:
            self._image = read_image(self._filepath)
            return self._image
    
    def clean(self):
        try:
            del self._rightframe
        except:
            pass
        try:
            del self._image
            del self._smoothed
        except:
            pass
    
    def get_width(self):
        return self.get_image().shape[1]
    
    def get_height(self):
        return self.get_image().shape[0]

    # blur the image to supress noise from being detected
    def get_smoothed(self):
        try:
            return self._smoothed
        except:
            self._smoothed = cv2.GaussianBlur(self.get_image(),(3,3),0)
        return self._smoothed
    
    # compute the median of the single channel pixel intensities for thresholding
    def get_median(self):
        try:
            return self._median
        except:
            self._median = np.median(self.get_image()) * 0.95
        return self._median

    def compute_depth(self):
        # find the disparity for all keypoints between the left and right image
        for kp in self.get_observations():
            kp.get_disparity(self.get_right_frame())
            
    def get_pose(self):
        return self._pose
    
    def set_pose(self, pose):
        self._pose = pose
    
    def get_pose_wrt(self, frame_origin):
        if self == frame_origin:
            return self.get_pose()
        else:
            return np.dot( self.keyframe.get_pose_wrt(frame_origin), self.get_pose() )
    
    def get_observations_xyz(self):
        return np.array([p.get_affine_coords() for p in self.get_observations() if p.get_mappoint() is None], dtype=np.float64, order='f')
    
    def get_observations_wc_np(self, frame_origin):
        try:
            return self.observations_wc
        except:
            self.observations_wc = np.dot( self.get_observations_xyz(), np.linalg.inv(self.get_pose_wrt(frame_origin)).T )
            return self.observations_wc
    
    def filter_observations(self, filter):
        self._observations = [obs for obs in self.get_observations() if filter(obs)]
    
    def filter_not_useful(self, stereo_confidence=STEREO_CONFIDENCE):
        self.filter_observations(lambda x: x.has_mappoint() or (x.disparity is not None and x.confidence > stereo_confidence))

    def filter_most_confident(self):
        self.filter_observations(lambda x: x.disparity is not None)
        self._observations.sort(key = lambda x: -x.confidence)
        seen = set()
        keep = []
        for o in self._observations:
            if (o.cx, o.cy) not in seen:
                keep.append(o)
        self._observations = keep
        
    def filter_non_mappoint(self):
        self.filter_observations(lambda x: x.has_mappoint())
        
    def get_observations(self):
        try:
            return self._observations
        except:
            zeroimage = zero_image(self.get_smoothed())
            #higher_vertical_edges = higher_vertical_edge(self.get_smoothed(), self.get_median())
            higher_vertical_edges = sobelv(self.get_smoothed(), self.get_median() * 1.1)
            #lower_vertical_edges = lower_vertical_edge(self.get_smoothed(), self.get_median())
            lower_vertical_edges = higher_vertical_edges

            veTop = top_vertical_edge(higher_vertical_edges)
            veBottom = bottom_vertical_edge(lower_vertical_edges)
            veBottom[-1:,:] = zeroimage[-1:,:]
            veBottom[:PATCH_SIZE,:] = zeroimage[0:PATCH_SIZE,:]
            veBottom[:,:PATCH_SIZE+2] = zeroimage[:,:PATCH_SIZE+2]
            veBottom[:,-PATCH_SIZE:] = zeroimage[:,-PATCH_SIZE:]
            veTop[-PATCH_SIZE:,:] = zeroimage[-PATCH_SIZE:,:]
            veTop[:1,:] = zeroimage[:1,:]
            veTop[:,:PATCH_SIZE+2] = zeroimage[:,:PATCH_SIZE+2]
            veTop[:,-PATCH_SIZE:] = zeroimage[:,-PATCH_SIZE:]
            
            # combine pixels found at the top and bottom of edges
            # results in an image where keypoints are set as pixels with a 255 intensity
            #keypointImage = veTop + veBottom

            #convert from pixels in an image to KeyPoints
            keypoints = np.column_stack(np.where(veTop >= 255))
            bottomleftpoints = [ObservationTopLeft(self, x, y) for y,x in keypoints]
            bottomrightpoints = [ObservationTopRight(self, x, y) for y,x in keypoints]
            keypoints = np.column_stack(np.where(veBottom >= 255))
            topleftpoints = [ObservationBottomLeft(self, x, y) for y,x in keypoints]
            toprightpoints = [ObservationBottomRight(self, x, y) for y,x in keypoints]
            self._observations = topleftpoints + toprightpoints + bottomleftpoints + bottomrightpoints
            return self._observations
