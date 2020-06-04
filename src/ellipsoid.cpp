#include "ellipsoid.hpp"

namespace obj_slam {

    ellipsoid::ellipsoid() {
        pose  = Sophus::SE3d();
        scale = Eigen::Vector3d::Ones();
    }

    ellipsoid::ellipsoid(const Sophus::SE3d& _p, const Eigen::Vector3d& _s) : 
        pose(_p), scale(_s) { }

    
    ellipsoid::ellipsoid(const Eigen::Vector9d& _tangent) {
        pose  = Sophus::SE3d::exp(_tangent.head<6>());
        scale = _tangent.tail<3>();
    }

    ellipsoid::ellipsoid(const Eigen::Vector10d& _dual) {
        Eigen::Matrix4d Q_star;
        Q_star << _dual[0], _dual[1], _dual[2], _dual[3],
                  _dual[1], _dual[4], _dual[5], _dual[6],
                  _dual[2], _dual[5], _dual[7], _dual[8],
                  _dual[3], _dual[6], _dual[8], _dual[9];
        Eigen::Matrix4d Q = Q_star.inverse() / std::cbrt(Q_star.determinant());
        Eigen::Matrix3d Q_33 = Q.block<3, 3>(0, 0);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Q_33);
        Eigen::Matrix3d eigen_vectors = solver.eigenvectors();
        Eigen::Vector3d eigen_values  = solver.eigenvalues();

        eigen_vectors = eigen_vectors / eigen_vectors.determinant();
        Eigen::Vector3d eigen_values_inv = eigen_values.array().inverse();

        scale = (-Q.determinant() / Q_33.determinant() * eigen_values_inv).array().sqrt();
        
        // check abs(eigen_vectors.det) == 1.
        assert(std::abs(1. - eigen_vectors.determinant()) < 1e-8);

        Sophus::SO3d so3(eigen_vectors / eigen_vectors.determinant());
        Eigen::Vector3d t(_dual[3], _dual[6], _dual[8]);
        t /= _dual[9];
        pose = Sophus::SE3d(so3, t);
    }

    ellipsoid ellipsoid::update(const Eigen::Vector9d& delta) const {
        Sophus::SE3d pose = Sophus::SE3d::exp(delta.head<6>()) * this->pose;
        Eigen::Vector3d scale = delta.tail<3>() + this->scale;
        return ellipsoid(pose, scale);
    }

    ellipsoid& ellipsoid::update_inplace(const Eigen::Vector9d& delta) {
        this->pose = Sophus::SE3d::exp(delta.head<6>()) * this->pose;
        this->scale += delta.tail<3>();
        return *this;
    }

    Eigen::Matrix4d ellipsoid::dual_mat() const {
        Eigen::Matrix4d std_form = Eigen::Matrix4d::Zero();

        std_form(0, 0) = scale[0] * scale[0];
        std_form(1, 1) = scale[1] * scale[1];
        std_form(2, 2) = scale[2] * scale[2];
        std_form(3, 3) = -1;

        Eigen::Matrix4d T = pose.matrix();
        return T * std_form * T.transpose();
    }

    Eigen::Matrix3d ellipsoid::conic_mat_on(
        const Sophus::SE3d& _cam_pose, const Eigen::Matrix3d& _cam_mat
    ) const {
        Eigen::Matrix<double, 3, 4> P = 
            _cam_mat * _cam_pose.matrix3x4();
        Eigen::Matrix4d Q_star = this->dual_mat();
        Eigen::Matrix3d C_star = P * Q_star * P.transpose();
        return C_star.adjoint();
    }

    Eigen::Vector4d ellipsoid::bbox_on(
        const Sophus::SE3d&    _cam_pose, 
        const Eigen::Matrix3d& _cam_mat, 
        int                    _height, 
        int                    _width
    ) const {
        Eigen::Matrix3d conic_mat = 
            this->conic_mat_on(_cam_pose, _cam_mat);

        std::vector<Eigen::Vector2d> candidates;
        this->_find_extream_pts(conic_mat, _height, _width, candidates);
        assert(!candidates.empty());

        double xmin = _width, ymin = _height;
        double xmax = -1.   , ymax = -1.;
        for (auto& each : candidates) {
            if (each.x() < xmin) { 
                xmin = each.x();
            }
            if (each.x() > xmax) {
                xmax = each.x();
            }
            if (each.y() < ymin) { 
                ymin = each.y();
            }
            if (each.y() > ymax) {
                ymax = each.y();
            }
        }

        return { xmin, ymin, xmax, ymax };
    }

    void ellipsoid::_find_extream_pts(
        const Eigen::Matrix3d&        conic_mat, 
        int                           h, 
        int                           w, 
        std::vector<Eigen::Vector2d>& candidates
    ) {
        double a = conic_mat(0, 0);
        double b = conic_mat(1, 1);
        double c = conic_mat(0, 1);
        double d = conic_mat(0, 2);
        double e = conic_mat(1, 2);
        double f = conic_mat(2, 2);

        candidates.clear();
        candidates.reserve(12);

        /**
         * C -> (a, c, d)
         *      (c, b, e)
         *      (d, e, f)
         * ax2 + by2 + 2cxy + 2dx + 2ey + f = 0
         */ 
        
        /**
         * dy/dx = 0 -> ax + cy + d = 0
         */ {
            double A = a * b - c * c;
            double B = 2. * (a * e - c * d);
            double C = a * f - d * d; 
            double y0, y1;
            bool ret = tools::solve_quadratic_equation(A, B, C, y0, y1);
            if (ret) {
                double x0 = -(c * y0 + d) / a;
                double x1 = -(c * y1 + d) / a;
                if (_in_viewport(x0, y0, h, w)) {
                    candidates.emplace_back(x0, y0);
                }
                if (_in_viewport(x1, y1, h, w)) {
                    candidates.emplace_back(x1, y1);
                }
            }
        }
        /**
         * dx/dy = 0 -> cx + by + e = 0
         */ {
            
            double A = a * b - c * c;
            double B = 2. * (b * d - c * e);
            double C = b * f - e * e;
            double x0, x1; 
            bool ret = tools::solve_quadratic_equation(A, B, C, x0, x1);
            if (ret) {
                double y0 = -(c * x0 + e) / b;
                double y1 = -(c * x1 + e) / b;
                if (_in_viewport(x0, y0, h, w)) {
                    candidates.emplace_back(x0, y0);
                }
                if (_in_viewport(x1, y1, h, w)) {
                    candidates.emplace_back(x1, y1);
                }
            }
        }

        /**
         * x = 0
         */ {
            double y0, y1;
            bool ret = tools::solve_quadratic_equation(b, 2. * e, f, y0, y1);
            if (ret) {
                if (_in_viewport(0, y0, h, w)) {
                    candidates.emplace_back(0, y0);
                }
                if (_in_viewport(0, y1, h, w)) {
                    candidates.emplace_back(0, y1);
                }
            }
        }
        /**
         * x = width - 1
         */ {
            double x = w - 1.;
            double A = b;
            double B = 2. * (c * x + e);
            double C = a * x * x + 2. * d * x + f;
            double y0, y1;
            bool ret = tools::solve_quadratic_equation(A, B, C, y0, y1);
            if (ret) {
                if (_in_viewport(x, y0, h, w)) {
                    candidates.emplace_back(x, y0);
                }
                if (_in_viewport(x, y1, h, w)) {
                    candidates.emplace_back(x, y1);
                }
            }
        }
        /**
         * y = 0
         */ {
            double x0, x1;
            bool ret = tools::solve_quadratic_equation(a, 2. * d, f, x0, x1);
            if (ret) {
                if (_in_viewport(x0, 0, h, w)) {
                    candidates.emplace_back(x0, 0);
                }
                if (_in_viewport(x1, 0, h, w)) {
                    candidates.emplace_back(x1, 0);
                }
            }
        }
        /**
         * y = height - 1
         */ {
            double y = h - 1.;
            double A = a;
            double B = 2. * (c * y + d);
            double C = b * y * y + 2. * e * y + f;
            double x0, x1;
            bool ret = tools::solve_quadratic_equation(A, B, C, x0, x1);
            if (ret) {
                if (_in_viewport(x0, y, h, w)) {
                    candidates.emplace_back(x0, y);
                }
                if (_in_viewport(x1, y, h, w)) {
                    candidates.emplace_back(x1, y);
                }
            }
        }
    }

}