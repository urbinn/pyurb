import os
os.environ['STEREO_CONFIDENCE'] = '1.6'
os.environ['SEQUENCE_CONFIDENCE'] = '1.6'
os.environ['PATCH_SIZE'] = '17'

import sys
SEQUENCE = '00' if len(sys.argv) < 2 else sys.argv[1]

from urb_kitti import *
import numpy as np
import glob

LEFTDIR = '/data/urbinn/datasets/kitti/sequences/%02d/image_2'%(int(SEQUENCE))
RIGHTDIR = '/data/urbinn/datasets/kitti/sequences/%02d/image_3'%(int(SEQUENCE))
OUTDIR = 'results8chi2'

FILES = len(list(glob.glob(LEFTDIR + '/*')))
FRAMECOUNT = FILES
#FRAMECOUNT = 250

seq = Sequence()
for frameid in range(FRAMECOUNT):
    if frameid % 100 == 0:
        print('frameid ', frameid)
    filename = '%06d.png'%(frameid)
    left_frame = Frame(LEFTDIR + '/' + filename, RIGHTDIR)
    seq.add_frame(left_frame)
    
keyframes_np = keyframes_to_np(seq.keyframes)
mappoints = get_mappoints(seq.keyframes)
mappoints_np = mappoints_to_np(mappoints)
links_np = links_to_np(mappoints)

if FRAMECOUNT == FILES:
    FRAMECOUNT = 'all'
suffix = '_{}_{}_{}_{}_{}'.format(SEQUENCE, FRAMECOUNT, PATCH_SIZE, STEREO_CONFIDENCE, SEQUENCE_CONFIDENCE)
np.save(OUTDIR + '/mappoints' + suffix, mappoints_np)
np.save(OUTDIR + '/links' + suffix, links_np)
np.save(OUTDIR + '/keyframes' + suffix, keyframes_np)
