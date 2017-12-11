#ifndef MOTION_ONLY_BA_H
#define MOTION_ONLY_BA_H

#include <Eigen/StdVector>

int LocalBundleAdjustment(Eigen::MatrixXf keyframes, Eigen::MatrixXf worldMapPoints, Eigen::MatrixXf keyframeWorldMapPoint);

#endif // MOTION_ONLY_BA_H
