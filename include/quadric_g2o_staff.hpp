#ifndef _QUADRIC_G2O_STAFF_HPP_
#define _QUADRIC_G2O_STAFF_HPP_

#include "common.hpp"
#include "ellipsoid.hpp"

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