#ifndef _ELLIPSOID_HPP_
#define _ELLIPSOID_HPP_

#include <sophus_templ/se3.hpp>

namespace Eigen {
    using Vector9d  = Matrix<double, 9, 1>;
    using Vector10d = Matrix<double, 10, 1>;
}

namespace tools {

    inline bool solve_quadratic_equation(
        double a, double b, double c, double& x0, double& x1
    ) {
        assert(0. != a);
        double delta = b * b - 4 * a * c;
        if (delta < 0.) { return false; }
        double delta_sqrt = std::sqrt(delta);
        x0 = (-b + delta_sqrt) / (2. * a);
        x1 = (-b - delta_sqrt) / (2. * a);
        return true;
    }
}

namespace obj_slam {

    struct ellipsoid {

        /**
         * @field pose the SE(3) transformation from world to object center
         * @field scale the 3-d scale of the object
         */ 
        Sophus::SE3d    pose;
        Eigen::Vector3d scale;

        ellipsoid();
        ellipsoid(const Sophus::SE3d& _p, const Eigen::Vector3d& _s);

        /**
         * @param _tangent (t0, t1, t2, r0, r1, r2, s0, s1, s2)
         */ 
        explicit ellipsoid(const Eigen::Vector9d& _tangent);

        /**
         * @param _dual the minimal representation of dual quadric
         */ 
        explicit ellipsoid(const Eigen::Vector10d& _dual);

        /**
         * @brief update the state of the object by left-multiplying delta
         * @param delta the updation (t0, t1, t2, r0, r1, r2, s0, s1, s2)
         */ 
        ellipsoid update(const Eigen::Vector9d& delta) const;
        ellipsoid& update_inplace(const Eigen::Vector9d& delta);

        Eigen::Matrix4d dual_mat() const;
        Eigen::Matrix3d conic_mat_on(const Sophus::SE3d& _cam_pose, const Eigen::Matrix3d& _cam_mat) const;
        Eigen::Vector4d bbox_on(const Sophus::SE3d& _cam_pose, const Eigen::Matrix3d& _cam_mat, int _height, int _width) const;

        static Eigen::Vector9d log(const ellipsoid& e);
        static ellipsoid exp(const Eigen::Vector9d& tangent);

    //private:
        static bool _in_viewport(double x, double y, int h, int w);
        static void _find_extream_pts(const Eigen::Matrix3d& conic_mat, int h, int w, std::vector<Eigen::Vector2d>& candidates);
    };

    inline Eigen::Vector9d ellipsoid::log(const ellipsoid& e) {
        Eigen::Vector9d tangent;
        tangent.head<6>() = e.pose.log();
        tangent.tail<3>() = e.scale;
        return tangent;
    }
    
    inline ellipsoid ellipsoid::exp(const Eigen::Vector9d& tangent) { 
        return ellipsoid(tangent); 
    }

    inline bool ellipsoid::_in_viewport(double x, double y, int h, int w) {
        return 0. <= x && x <= w - 1 && 0. <= y && y <= h - 1;
    }
}

#endif