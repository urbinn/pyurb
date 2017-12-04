from settings.load import *
from urb_frame import *
from urb_framepoint import *
from urb_json import *
import sys
import os
import shutil
import urbg2o
import numpy as np

def save_keyframepoints(filename, keyFramePoints):
    with open(filename, 'w') as fout:
        for p in keyFramePoints:
            fout.write('%d %d %d %f\n'%(p.id, p.cx, p.cy, p.get_depth()))

def save_framepoints(filename, frame_points):
    with open(filename, 'w') as fout:
        for p in frame_points:
            if p.matches is not None and p.matches.get_depth() is not None:
                fout.write('%d %d %d %f %d %d\n'%(p.matches.id, p.matches.cx, p.matches.cy, p.matches.get_depth(), p.cx, p.cy))

# returns the two best matching points in the list of keyPoints for the given query keyPoint
def matching_framepoint(frame_point, frame_points):
    if len(frame_points) < 2:
         return (0, None)
    best_distance = sys.maxsize
    next_best_distance = sys.maxsize
    best_frame_point = None
    for kp in frame_points:
        distance = kp.get_patch_distance(frame_point)
        if distance < best_distance:
            next_best_distance = best_distance
            best_distance = distance
            best_frame_point = kp
        elif distance < next_best_distance:
            next_best_distance = distance
    confidence = next_best_distance / (best_distance + 0.01)
    return (confidence, best_frame_point)

# returns the matching keyPoints in a new frame to keyPoints in a keyFrame that exceed a confidence score
def match_frame(frame, frame_points, confidence_threshold = 1.4):
    for kp in frame_points:
        confidence, fp = matching_framepoint(kp, frame.get_framepoints())
        if confidence > confidence_threshold and fp.matches is None:
            fp.matches = kp
            
def create_sequence(frames):
    s = Sequence()
    for i, f in enumerate(frames):
        print('add frame ' + str(i))
        s.add_frame(f);
    return s

class Sequence:
    def __init__(self):
        self.framepointcount = 0
        self.keyframes = []

    def add_frame(self, frame, confidence_threshold = 1.4, clean=False):
        if len(self.keyframes) == 0:
            frame.compute_depth()
            frame.filter_non_stereo()
            self.add_keyframe(frame)
            frame.set_pose( np.eye(4, dtype=np.float64) )
        else:
            keyframe = self.keyframes[-1]
            match_frame(frame, keyframe.get_framepoints(), confidence_threshold = confidence_threshold)
            #print(frame.get_pose())
            # make the former frame into a keyframe
            if frame.get_pose() is None:
                if clean:
                    for f in keyframe.frames[:-1]:
                        f.clean()
                    keyframe.clean()
                keyframe = keyframe.frames.pop()
                keyframe.compute_depth()
                keyframe.filter_non_stereo()
                self.add_keyframe( keyframe )
                match_frame(frame, keyframe.get_framepoints(), confidence_threshold = confidence_threshold)
            frame.keyframe = keyframe
            keyframe.frames.append(frame)
                
    def add_keyframe(self, frame):
        for fp in frame.get_framepoints():
            if fp.matches is None:
                self.framepointcount += 1
                fp.id = self.framepointcount
        self.keyframes.append(frame)
        frame.frames = []
    
    def keyframes_to_np(self):
        kf = np.vstack([kf.get_pose().flatten() for kf in self.keyframes])
        return kf
    
    def keyframepoints_to_np(self):
        kfp = np.array([p.get_affine_coords() for f in self.keyframes for p in f.get_framepoints() if p.matches == None], dtype=np.float64, order='f')
        return kfp
                        
    def links_to_np(self):
        links = [(i, p.id, p.cx, p.cy) for i, f in enumerate(self.keyframes) for p in f.get_framepoints() if p.id is not None]
        return np.array(links, dtype=np.int32, order='f')
        
    def dump(self, folder):
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)
        #save_keyframepoints(folder + '/0.txt', self.get_keyframepoints())
        for i, frame in enumerate(self.frames):
            save_framepoints(folder + '/' + str(i+1) + '.txt', frame.get_framepoints())
                
    def counts(self):
        r = [0] * len(self.get_keyframepoints())
        for f in self.frames:
            for p in f.get_framepoints():
                r[p.id] += 1
        return r