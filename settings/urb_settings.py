import cv2

# Set constants
# de breedte en hoogte van de patches die gematched worden
# een lagere patchSize resulteert in meer punten
NORM = cv2.NORM_L1
PATCH_SIZE = 17
HALF_PATCH_SIZE = PATCH_SIZE // 2
MONO_PATCH_SIZE = 9
MONO_HALF_PATCH_SIZE = MONO_PATCH_SIZE // 2
CONFIDENCE = 1.8
