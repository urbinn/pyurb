import cv2
import os

# Set constants
# de breedte en hoogte van de patches die gematched worden
# een lagere patchSize resulteert in meer punten

def env_int(variable, default):
    return int(os.environ[variable]) if variable in os.environ else default

def env_float(variable, default):
    return float(os.environ[variable]) if variable in os.environ else default


NORM = cv2.NORM_L1
PATCH_SIZE = env_int('PATCH_SIZE', 17)
HALF_PATCH_SIZE = PATCH_SIZE // 2    
STEREO_CONFIDENCE = env_float('STEREO_CONFIDENCE', 1.6)
SEQUENCE_CONFIDENCE = env_float('SEQUENCE_CONFIDENCE', 1.6)
