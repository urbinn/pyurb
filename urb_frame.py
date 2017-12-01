from collections import defaultdict
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from urb_filter import *
from urb_framepoint import *
from urb_imageio import *
from urb_constants import *
import urbg2o
#from numpyctypes import c_ndarray

def framepoints_to_numpy(frame_points):
    fps = [(fp.matches.id, fp.matches.cx, fp.matches.cy, fp.matches.get_depth(), fp.cx, fp.cy) for fp in frame_points if fp.matches is not None]
    arr = np.array(fps, dtype=np.float64, order='f')
    return arr

class Frame:
    def __init__(self, filepath, rightpath = None):
        self.filepath = filepath
        self.image = read_image(filepath)
        self.pose = None
        self.keyframe = None
        self.rightpath = rightpath
        
    def get_right_frame(self):
        try:
            return self.rightframe
        except:
            if self.rightpath is None:
                raise ValueError('rightpath is not set')
            self.rightframe = Frame('/'.join([self.rightpath, self.filepath.split('/')[-1]]))
            return self.rightframe
        
    def get_image(self):
        return self.image
    
    def get_width(self):
        return self.image.shape[1]
    
    def get_height(self):
        return self.image.shape[0]

    # blur the image to supress noise from being detected
    def get_smoothed(self):
        try:
            return self.smoothed
        except:
            self.smoothed = cv2.GaussianBlur(self.image,(5,5),0)
        return self.smoothed
    
    # compute the median of the single channel pixel intensities for thresholding
    def get_median(self):
        try:
            return self.median
        except:
            self.median = np.median(self.image) * 0.95
        return self.median

    def compute_depth(self):
        # find the disparity for all keypoints between the left and right image
        for kp in self.framepoints:
            kp.get_disparity(self.get_right_frame())
            
    def get_pose(self):
        if self.pose is None:
            pose = np.ndarray((4,4), dtype=np.float64, order='f')
            fps = framepoints_to_numpy(self.get_framepoints())
            pointsLeft = urbg2o.poseOptimization(fps, pose)
            if pointsLeft > 10:
                self.pose = pose
        return self.pose
    
    def get_pose_wrt(self, frame_origin):
        if self == frame_origin:
            return self.get_pose()
        else:
            return np.dot( self.keyframe.get_pose(), self.get_pose() )
    
    def get_framepoints_xyz(self):
        return np.array([p.get_affine_coords() for p in self.get_framepoints() if p.matches == None], dtype=np.float64, order='f')
    
    def get_framepoints_wc(self, frame_origin):
        try:
            return self.framepoints_wc
        except:
            self.framepoints_wc = np.dot( self.get_framepoints_xyz(), np.linalg.inv(self.get_pose_wrt(frame_origin)).T )
            return self.framepoints_wc
    
    def filter_non_stereo(self, confidence=CONFIDENCE):
        self.framepoints = [fp for fp in self.framepoints if fp.disparity is not None and fp.confidence > confidence]

    def filter_non_id(self):
        self.framepoints = [fp for fp in self.framepoints if fp.id is not None]

    def filterNN(self,):
        self.framepoints = self.filterNN(self.framepoints)
        
    def get_framepoints(self):
        try:
            return self.framepoints
        except:
            zeroimage = zero_image(self.get_smoothed())
            #higher_vertical_edges = higher_vertical_edge(self.get_smoothed(), self.get_median())
            higher_vertical_edges = sobelv(self.get_smoothed(), self.get_median() * 1.1)
            #lower_vertical_edges = lower_vertical_edge(self.get_smoothed(), self.get_median())
            lower_vertical_edges = higher_vertical_edges

            veTop = top_vertical_edge(higher_vertical_edges)
            veBottom = bottom_vertical_edge(lower_vertical_edges)
            veTop[0:PATCH_SIZE,:] = zeroimage[0:PATCH_SIZE,:]
            veTop[:,:HALF_PATCH_SIZE] = zeroimage[:,:HALF_PATCH_SIZE]
            veTop[:,-HALF_PATCH_SIZE:] = zeroimage[:,-HALF_PATCH_SIZE:]
            veBottom[-PATCH_SIZE:,:] = zeroimage[-PATCH_SIZE:,:]
            veBottom[:,:HALF_PATCH_SIZE] = zeroimage[:,:HALF_PATCH_SIZE]
            veBottom[:,-HALF_PATCH_SIZE:] = zeroimage[:,-HALF_PATCH_SIZE:]
            
            # combine pixels found at the top and bottom of edges
            # results in an image where keypoints are set as pixels with a 255 intensity
            #keypointImage = veTop + veBottom

            #convert from pixels in an image to KeyPoints
            keypoints = np.column_stack(np.where(veTop >= 255))
            toppoints = [FramePointBottom(self, x, y) for y,x in keypoints]
            keypoints = np.column_stack(np.where(veBottom >= 255))
            bottompoints = [FramePointTop(self, x, y) for y,x in keypoints]
            self.framepoints = toppoints + bottompoints
            for i,f in enumerate(self.framepoints):
                f.id = i
            return self.framepoints

   