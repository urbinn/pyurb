from settings.load import *
from urb_frame import *
from urb_json import *
import sys
import os
import shutil
import urbg2o
import numpy as np

# returns the two best matching points in the list of keyPoints for the given query keyPoint
def matching_framepoint(o, observations):
    if len(observations) < 2:
         return (0, None)
    best_distance = sys.maxsize
    next_best_distance = sys.maxsize
    best_frame_point = None
    for kp in observations:
        distance = kp.get_patch_distance(o)
        if distance < best_distance:
            next_best_distance = best_distance
            best_distance = distance
            best_frame_point = kp
        elif distance < next_best_distance:
            next_best_distance = distance
    confidence = next_best_distance / (best_distance + 0.01)
    return (confidence, best_frame_point)

# returns the matching keyPoints in a new frame to keyPoints in a keyFrame that exceed a confidence score
def match_frame(frame, observations, confidence_threshold = 1.4):
    for obs in observations:
        confidence, fp = matching_framepoint(obs, frame.get_observations())
        if confidence > confidence_threshold:
            fp.set_mappoint(obs.get_mappoint())
        else:
            fp.set_mappoint(None)
            
def create_sequence(frames):
    s = Sequence()
    for i, f in enumerate(frames):
        print('add frame ' + str(i))
        s.add_frame(f);
    return s

class Sequence:
    def __init__(self):
        self.mappointcount = 0
        self.framecount = 0
        self.keyframes = []

    def add_frame(self, frame, confidence_threshold = 1.4, clean=False):
        if len(self.keyframes) == 0:
            self.add_keyframe(frame)
            frame.set_pose( np.eye(4, dtype=np.float64) )
        else:
            keyframe = self.keyframes[-1]
            match_frame(frame, keyframe.get_observations(), confidence_threshold = confidence_threshold)
            # make the former frame into a keyframe
            if frame.get_pose() is None:
                if clean:
                    for f in keyframe.frames[:-1]:
                        f.clean()
                    keyframe.clean()
                keyframe = keyframe.frames.pop()
                self.add_keyframe( keyframe )
                match_frame(frame, keyframe.get_observations(), confidence_threshold = confidence_threshold)
                if frame.get_pose() is None:
                    print('WARNING: invalid pose estimation')
            frame.keyframe = keyframe
            keyframe.frames.append(frame)
                
    def add_keyframe(self, frame):
        frame.compute_depth()
        frame.filter_not_useful()
        for obs in frame.get_observations():
            if not obs.has_mappoint():
                obs.create_mappoint(self.mappointcount)
                self.mappointcount += 1
        frame.id = self.framecount
        self.framecount += 1
        frame.register_mappoints()
        self.keyframes.append(frame)
        frame.frames = []
            
    def dump(self, folder):
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)
        #save_keyframepoints(folder + '/0.txt', self.get_keyframepoints())
        for i, frame in enumerate(self.frames):
            save_framepoints(folder + '/' + str(i+1) + '.txt', frame.get_observations())

def get_covisible_keyframes(keyframe):
    mappoints = [ o.get_mappoint() for o in keyframe.get_observations() ]
    return { o.get_frame() for m in mappoints for o in m.get_observations() }

def get_mappoints(keyframes):
    return { o.get_mappoint() for kf in keyframes for o in kf.get_observations() }

def get_fixed_keyframes(mappoints, covisible_keyframes):
    frames = { o.get_frame() for m in mappoints for o in m.get_observations() }
    return frames - covisible_keyframes

def keyframes_to_np(keyframes):
    return np.hstack([ [ [kf.id] for kf in keyframes ], [ kf.get_pose().flatten() for kf in keyframes ] ])

def mappoints_to_np(mappoints):
    kfp = np.hstack([ [ [m.id] for m in mappoints ], [ m.get_affine_coords() for m in mappoints ] ])
    return kfp

def links_to_np(mappoints):
    links = [ (m.id, o.get_frame().id, o.cx, o.cy) for m in mappoints for o in m.get_observations()]
    return np.array(links, dtype=np.float64, order='f')

