#ifndef _QUADRIC_G2O_STAFF_HPP_
#define _QUADRIC_G2O_STAFF_HPP_

#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

#include "ellipsoid.hpp"

namespace tools {

    inline Sophus::SE3d to_sophus(const g2o::SE3Quat& pose) {
        return Sophus::SE3d(pose.rotation(), pose.translation());
    }

    /**
     * @brief calculate the overlap of the [a0, a1] with [b0, b1]
     */ 
    inline double overlap(double a0, double a1, double b0, double b1) {
        double left  = a0 > b0 ? a0 : b0;
        double right = a1 < b1 ? a1 : b1;
        return right - left;  
    }

    /**
     * @brief calculate the box_intersection of two boxes
     * @note box represented as (xmin, ymin, xmax, ymax)
     */
    inline double box_intersection(
        const Eigen::Vector4d& box0, 
        const Eigen::Vector4d& box1
    ) {
        double w = overlap(box0[0], box0[2], box1[0], box1[2]);
        double h = overlap(box0[1], box0[3], box1[1], box1[3]);
        return w * h;
    }

    inline double box_area(const Eigen::Vector4d& box) {
        return (box[2] - box[0]) * (box[3] - box[1]);
    }

    inline double box_iou(
        const Eigen::Vector4d& box0, 
        const Eigen::Vector4d& box1
    ) {
        double i = box_intersection(box0, box1);
        double u = box_area(box0) + box_area(box1) - i;
        return i / u;
    }
}

namespace g2o {

    struct vertex_quadric : 
        BaseVertex<9, obj_slam::ellipsoid> {

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        vertex_quadric() = default;

        bool read(std::istream& is) override { return true; }
        bool write(std::ostream& os) const override { return true; }

        void setToOriginImpl() override { _estimate = obj_slam::ellipsoid(); }

        void oplusImpl(const double* u) override {
            Eigen::Map<const Eigen::Vector9d> delta(u);
            _estimate.update_inplace(delta);
        }
    };

    struct edge_quadric_proj : 
        BaseBinaryEdge<4, Vector4d, VertexSE3Expmap, vertex_quadric> {
        
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        edge_quadric_proj(float fx, float fy, float cx, float cy, int h, int w);

        bool read(std::istream& is) override { return true; }
        bool write(std::ostream& os) const override { return true; }

        void computeError() override;

        Eigen::Matrix3d cam_mat;
        int width, height;
    };

    struct edge_quadric_proj_iou : 
        BaseBinaryEdge<1, Vector4d, VertexSE3Expmap, vertex_quadric> {
        
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        edge_quadric_proj_iou(float fx, float fy, float cx, float cy, int h, int w);

        bool read(std::istream& is) override { return true; }
        bool write(std::ostream& os) const override { return true; }

        void computeError() override;

        Eigen::Matrix3d cam_mat;
        int width, height;
    };
    
} 

#endif