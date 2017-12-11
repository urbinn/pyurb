from progressbar import ProgressBar
from pyurb.settings.load import *
from pyurb.urb_frame import *
from pyurb.urb_json import *
import sys
import os
import shutil
import pyurb.urbg2o as urbg2o
import numpy as np

# returns the two best matching points in the list of keyPoints for the given query keyPoint
def matching_framepoint(o, observations):
    if len(observations) < 2:
         return (0, None)
    best_distance = sys.maxsize
    next_best_distance = sys.maxsize
    best_observation = None
    for kp in observations:
        if type(kp) is type(o):
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
def match_frame(frame, observations, sequence_confidence = SEQUENCE_CONFIDENCE):
    matches = []
    for i, obs in enumerate(observations):
        confidence, fp = matching_framepoint(obs, frame.get_observations())
        if confidence > sequence_confidence:
            matches.append(fp)
            fp.set_mappoint(obs.get_mappoint())
        else:
            fp.set_mappoint(None)
    return matches
            
def create_sequence(frames, sequence_confidence=SEQUENCE_CONFIDENCE):
    s = Sequence()
    for i, f in enumerate(ProgressBar()(frames)):
        #print('add frame ' + str(i))
        s.add_frame(f, sequence_confidence = sequence_confidence );
    return s

class Sequence:
    def __init__(self):
        self.mappointcount = 0
        self.framecount = 0
        self.keyframes = []
        self.rotation = 0
        self.speed = 0
        
    def add_frame(self, frame, sequence_confidence = SEQUENCE_CONFIDENCE, clean=False):
        if len(self.keyframes) == 0:
            self.add_keyframe(frame)
            frame.set_pose( np.eye(4, dtype=np.float64) )
        else:
            keyframe = self.keyframes[-1]
            last_z = keyframe.frames[-1].get_pose()[2, 3] if len(keyframe.frames) > 0 else 0
            matches = match_frame(frame, keyframe.get_observations(), sequence_confidence = sequence_confidence)
            
            points_left = len(matches)
            if points_left >=10:
                pose, points_left = get_pose(matches)
                frame.set_pose(pose)
                if len(self.keyframes) == 1 and len(keyframe.frames) == 0:
                    self.speed = pose[2,3]
                    self.rotation = pose[0,2]
                invalid_rotation= abs(self.rotation - pose[0,2]) > 0.2
                speed = (pose[2,3] - last_z)
                invalid_speed = speed < -2 or speed > 0
                if points_left >= 10:
                    if invalid_rotation:
                        print('invalid rotation', self.rotation, '\n', pose)
                    if invalid_speed:
                        print('invalid speed', speed, self.speed, '\n', pose)

            #print(len(matches), frame.get_pose())
            # make the former frame into a keyframe
            if points_left < 10 or invalid_rotation or invalid_speed:
                if clean:
                    for f in keyframe.frames[:-1]:
                        f.clean()
                    keyframe.clean()
                keyframe = keyframe.frames.pop()
                self.add_keyframe( keyframe )
                last_z = keyframe.frames[-1].get_pose()[2, 3] if len(keyframe.frames) > 0 else 0
                matches = match_frame(frame, keyframe.get_observations(), sequence_confidence = sequence_confidence)
                pose, points_left = get_pose(matches)
                invalid_rotation= abs(self.rotation - pose[0,2]) > 0.2
                speed = (pose[2,3] - last_z)
                invalid_speed = speed < -2 or speed > 0
                
                if points_left < 10 or invalid_rotation or invalid_speed:
                    print('WARNING: invalid pose estimation')
                    print(len(matches), speed, pose)
                    frame.set_pose(np.eye(4, dtype=np.float64))
                else:
                    frame.set_pose(pose) 

            self.speed = (self.speed + self.speed + (pose[2,3] - last_z))/3
            self.rotation = (self.rotation + self.rotation + pose[0,2])/3
            frame.keyframe = keyframe
            keyframe.frames.append(frame)
                
    def add_keyframe(self, frame):
        frame.compute_depth()
        frame.filter_not_useful()
        frame.filter_most_confident()
        for obs in frame.get_observations():
            if not obs.has_mappoint():
                obs.create_mappoint(self.mappointcount)
                self.mappointcount += 1
            else:
                obs.register_mappoint()
                if obs.get_depth() is None:
                    acoords = frame.get_pose() * obs.get_mappoint().get_affine_coords()
                    obs.z = affine_coords_to_cam(acoords)
                else:
                    obs.get_mappoint().update_affine_coords(obs)
        frame.id = self.framecount
        self.framecount += 1
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
    return { o.get_mappoint() for kf in keyframes for o in kf.get_observations() if o.get_mappoint() is not None }

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

