import json
import json_tricks

def loadi(i):
    try:
        return int(i)
    except:
        return None

def loadf(i):
    try:
        return float(i)
    except:
        return None
    
def dumpstr(i):
    if i is None:
        return None
    return str(i)

def loadnps(o):
    return json_tricks.loads(o)

def dumpnps(o):
    return json_tricks.dumps(o)
    
def saves(obj):
    return json.dumps(obj, indent=4, cls=UrbEncoder)

def loads(obj):
    return json.loads(obj, cls=UrbDecoder)
    
class UrbEncoder(json.JSONEncoder):
    def default(self, v):
        otype = type(v).__name__
        if otype == 'Frame':
            return frame_dumps(v)
        elif otype == 'FramePointTop':
            return framepointtop_dumps(v)
        elif otype == 'FramePointBottom':
            return framepointbottom_dumps(v)
        else:
            return json.JSONEncoder.default(self, v)
        
class UrbDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if '_type' not in obj:
            return obj
        otype = obj['_type']
        if type == 'Frame':
            return frame_loads(obj)
        elif type == 'FramePointTop':
            return framepointtop_loads(obj)
        elif type == 'FramePointBottom':
            return framepointbottom_loads(obj)
        return obj

def frame_loads(o):
    f = Frame(o['filepath'])
    f.framepoints = o['framepoints']
    f.rightpath = o['rightpath']
    f.pose = loadnps(o['pose'])
    return f

def frame_dumps(f):
    return { '_type': 'Frame', 'filepath': f._filepath, 'framepoints': f.get_framepoints(), 'rightpath': f._rightpath, 'pose': dumpnps(f._pose) }

def framepointbottom_dumps(fp):
        return { '_type': 'FramePointTop',  'id': dumpstr(fp.id), 'cx': str(fp.cx), 'cy': str(fp.cy), 
                'matches': dumps_attr2(fp, 'matches', 'id'), 'z': dumpstr(fp.z), 'disparity': dumpstr(fp.disparity) }

def framepointtop_dumps(fp):
    return { '_type': 'FramePointBottom', 'id': dumpstr(fp.id), 'cx': str(fp.cx), 'cy': str(fp.cy), 
            'matches': dumps_attr2(fp, 'matches', 'id'), 'z': dumpstr(fp.z), 'disparity': dumpstr(fp.disparity) }
    
def framepointtop_loads(o):
    f = FramePointTop(None, int(o['cx']), int(o['cy']))
    framepoint_loads(f, o)
    return f
   
def framepointbottom_loads(o):
    f = FramePointBottom(None, int(o['cx']), int(o['cy']))
    framepoint_loads(f, o)
    return f

def framepoint_loads(frame, o):
    frame.id = loadi(o['id'])
    frame.matches = loadi(o['matches'])
    frame.z = loadf(o['z'])
    frame.disparity = loadf(o['disparity'])
     
def dumps_attr(obj, attr):
    try:
        v = getattr(obj, attr)
        return dumpstr(v)
    except AttributeError:
        pass
    return None

def dumps_attr2(obj, attr, attr2):
    try:
        v = getattr(obj, attr)
        if v is not None:
            v2 = getattr(v, attr2)
            return dumpstr(v2)
    except AttributeError:
        pass
    return None




