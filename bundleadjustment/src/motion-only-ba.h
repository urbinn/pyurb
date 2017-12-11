#ifndef MOTION_ONLY_BA_H
#define MOTION_ONLY_BA_H

#include <Eigen/StdVector>

int poseOptimization(Eigen::Ref<Eigen::MatrixXd> coords, Eigen::Ref<Eigen::MatrixXd> pose);

#endif // MOTION_ONLY_BA_H
