#include <iostream>

#include "ellipsoid.hpp"

int main(int argc, char** argv) {

    Eigen::Matrix3d C = Eigen::Matrix3d::Identity();
    C(2, 2) = 100 * 100 * 2 - 142 * 142;
    C(0, 2) = -100;
    C(2, 0) = -100;
    C(1, 2) = -100;
    C(2, 1) = -100;

    std::vector<Eigen::Vector2d> candidates;
    obj_slam::ellipsoid::_find_extream_pts(C, 200, 300, candidates);

    for (auto& each : candidates) {
        std::cout << each.transpose() << std::endl;
    }

    return 0;
}