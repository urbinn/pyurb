from settings.load import *
from urb_framepoint import *
from urb_imageio import *
import cv2
import numpy as np
from urb_filter import *
import urbg2o

def framepoints_to_numpy(frame_points):
    fps = [(fp.matches.id, fp.matches.cx, fp.matches.cy, fp.matches.get_depth(), fp.cx, fp.cy) for fp in frame_points if fp.matches is not None]
    arr = np.array(fps, dtype=np.float64, order='f')
    return arr

class Frame:
    def __init__(self, filepath, rightpath = None):
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
            self._smoothed = cv2.GaussianBlur(self.get_image(),(5,5),0)
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
        for kp in self.get_framepoints():
            kp.get_disparity(self.get_right_frame())
            
    def get_pose(self):
        if self._pose is None:
            pose = np.ndarray((4,4), dtype=np.float64, order='f')
            fps = framepoints_to_numpy(self.get_framepoints())
            pointsLeft = urbg2o.poseOptimization(fps, pose)
            if pointsLeft > 10:
                self._pose = pose
        return self._pose
    
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
        self._framepoints = [fp for fp in self._framepoints if fp.disparity is not None and fp.confidence > confidence]

    def filter_non_id(self):
        self._framepoints = [fp for fp in self._framepoints if fp.id is not None]
        
    def get_framepoints(self):
        try:
            return self._framepoints
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
            self._framepoints = toppoints + bottompoints
            return self._framepoints
