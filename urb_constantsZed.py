import cv2

# de breedte en hoogte van de patches die gematched worden
# een lagere patchSize resulteert in meer punten
NORM = cv2.NORM_L1
#PATCH_SIZE = 9
PATCH_SIZE = 7
HALF_PATCH_SIZE = PATCH_SIZE // 2
MONO_PATCH_SIZE = 9
MONO_HALF_PATCH_SIZE = MONO_PATCH_SIZE // 2
CONFIDENCE = 1.8

###ZED
CAMERA_BF=0.12 * 700.726
CAMERA_FX =  700.726
CAMERA_FY =  700.726
CAMERA_CX = 679.831
CAMERA_CY = 370.07

CAMERA_FX_INV = 1.0 / CAMERA_FX;
CAMERA_FY_INV = 1.0 / CAMERA_FY;

