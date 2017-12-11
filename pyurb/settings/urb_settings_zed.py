from pyurb.settings.urb_settings import *

# bf is baseline * focal distance om de diepte te berekenen met formule focal distance * baseline / disparity
###ZED
CAMERA_BF=0.12 * 700.726
CAMERA_FX =  700.726
CAMERA_FY =  700.726
CAMERA_CX = 679.831
CAMERA_CY = 370.07

CAMERA_FX_INV = 1.0 / CAMERA_FX;
CAMERA_FY_INV = 1.0 / CAMERA_FY;
