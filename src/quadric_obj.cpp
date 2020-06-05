#include "quadric_obj.hpp"
#include "quadric_frame.hpp"

namespace obj_slam {

    size_t quadric_obj::_seq_id = 0;

    quadric_obj::quadric_obj() : 
        id(_seq_id++), state(NEED_OBSERVATIONS) { }

    void quadric_obj::observe(quadric_frame* frame, size_t box_idx) {
        auto ret = observations.insert(std::make_pair(frame, box_idx));
        assert(ret.second);
        if (3 == observations.size()) {
            param = _from_observations();
            state = UPDATING;
        }
    }

    ellipsoid quadric_obj::_from_observations() const {
        
        std::vector<Eigen::Vector4d> planes;
        planes.reserve(observations.size() * 4);

        for (auto& ob : observations) {

            const auto* frame = ob.first;
            const Eigen::Vector4d& bbox = frame->bboxes[ob.second].bbox;

            // plane = (KT)' * line
            Eigen::Matrix<double, 4, 3> KT_t = 
                (frame->cam_mat * frame->t_cw.matrix3x4()).transpose();

            planes.emplace_back(KT_t * Eigen::Vector3d(1, 0, -bbox[0]));
            planes.emplace_back(KT_t * Eigen::Vector3d(0, 1, -bbox[1]));
            planes.emplace_back(KT_t * Eigen::Vector3d(1, 0, -bbox[2]));
            planes.emplace_back(KT_t * Eigen::Vector3d(0, 1, -bbox[3]));
        }

        Eigen::MatrixXd A(planes.size(), 10);

        for (size_t i = 0; i < planes.size(); ++i) {
            const auto& plane = planes[i];

            double pi1 = plane[0];
            double pi2 = plane[1];
            double pi3 = plane[2];
            double pi4 = plane[3];

            // Aj
            A(i, 0) = pi1 * pi1;
            A(i, 1) = 2. * pi1 * pi2;
            A(i, 2) = 2. * pi1 * pi3;
            A(i, 3) = 2. * pi1 * pi4;
            A(i, 4) = pi2 * pi2;
            A(i, 5) = 2. * pi2 * pi3;
            A(i, 6) = 2. * pi2 * pi4;
            A(i, 7) = pi3 * pi3;
            A(i, 8) = 2. * pi3 * pi4;
            A(i, 9) = pi4 * pi4;
        }

        /**
         * @note A least squares solution q(minimal of Q*) that minimizes ||Aq|| 
         *       can be obtained as the last colunm of V, which Aq = UDV'
         * @cite QUADRICSLAM: DUAL QUADRICS FROM OBJECT DETECTIONS AS LANDMARKS 
         */
        Eigen::JacobiSVD<Eigen::MatrixXd> solver(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector10d q = solver.matrixV().col(9);
        return ellipsoid(q);
    }
}