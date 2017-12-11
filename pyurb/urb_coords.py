import sys
from pyurb.settings.load import *

def cam_to_affine_coords(u, v, z):
    return (u-CAMERA_CX) * z * CAMERA_FX_INV, (v-CAMERA_CY) * z * CAMERA_FY_INV, z, 1

def affine_coords_to_cam(coords):
    z = coords[2]
    x = coords[0] * CAMERA_FX / z + CAMERA_CX
    y = coords[1] * CAMERA_FY / z + CAMERA_CY
    return x, y, z

def estimated_distance(disparity):
    return -CAMERA_BF / disparity

# Estimates the subpixel disparity based on a parabola fitting of the three points around the minimum.
def subpixel_disparity(disparity , coords):
    try:
        subdisparity =  (coords[0] - coords[2]) / (2.0 * (coords[0] + coords[2] - 2.0 * coords[1]))
        return -max(disparity + subdisparity, 0.01)
    except:
        return -disparity - 1

#Compute the distance for a patch in the left hand image by computing the disparity of the patch 
# in the right hand image. A low confidence (near 1) indicates there is another location where the 
# patch also fits, and therefore the depth estimate may be wrong. 
# The bestDistance is returned along with its nearest neighbors to facilitate subpixel disparity estimation.
def patch_disparity(obs, frame_right):
    frame_left = obs.get_frame()
    if obs.cy < PATCH_SIZE or obs.cy > frame_left.get_height() - PATCH_SIZE or \
        obs.cx < PATCH_SIZE or obs.cx > frame_left.get_width() - PATCH_SIZE:
            return None, None
    best_disparity = 0
    best_distance = sys.maxsize
    distances = []
    for disparity in range(0, obs.leftx):

        patchL = obs.get_patch()
        patchR = frame_right.get_image()[obs.topy:obs.topy+PATCH_SIZE, 
                                                               obs.leftx-disparity:obs.leftx+PATCH_SIZE-disparity]
        #print(patchL.shape, patchR.shape,leftxstart,patchSize, disparity)
        distance = cv2.norm(patchL, patchR, NORM)
        distances.append(distance)
        if distance < best_distance:
            best_distance = distance
            best_disparity = disparity
    
    # bepaal minimale distance op disparities meer dan 1 pixel van lokale optimum 
    minrest = sys.maxsize
    if best_disparity > 1:
        minrest = min(distances[0:best_disparity-1])
    if best_disparity < obs.leftx - HALF_PATCH_SIZE - 2:
        minrest = min([minrest, min(distances[best_disparity+2:])])
        
    # de disparity schatting is onbetrouwbaar als die dicht bij 1 komt
    # gebruik hier als threshold 1.4 om punten eruit te filteren waarvoor we
    # geen betrouwbare disparity estimates kunnen maken
    confidence = minrest / (best_distance + 0.01)
    
    # Geef de beste disparity op pixel niveau terug, met de twee neighbors om subpixel disparity uit te rekenen
    if best_disparity == 0:
        disparity = subpixel_disparity(best_disparity, [best_distance, distances[1], distances[2]])
    elif best_disparity == obs.leftx -1:
        disparity = subpixel_disparity(best_disparity, [distances[best_disparity-2], distances[best_disparity-1], best_distance])
    else:
        disparity = subpixel_disparity(best_disparity, [distances[best_disparity-1], best_distance, distances[best_disparity+1]])
    return confidence, disparity
