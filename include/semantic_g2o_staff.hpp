#ifndef _SEMANTIC_G2O_STAFF_HPP_
#define _SEMANTIC_G2O_STAFF_HPP_

#include "ThirdParty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

namespace vso {

    struct semantic_lab;

    struct edge_semantic_err : 
        g2o::BaseBinaryEdge<
            1, semantic_lab*, g2o::VertexSE3Expmap, g2o::VertexSBAPointXYZ
        > 
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        edge_semantic_err(float _fx, float _fy, float _cx, float _cy, const float* _w) :
            fx(_fx), fy(_fy), cx(_cx), cy(_cy), w(_w) { } 

        bool edge_semantic_err::read(std::istream& is) override { return true; }
        bool edge_semantic_err::write(std::ostream& os) const override { return true; }

        void computeError() override;

        Eigen::Vector2d cam2pixel(const Eigen::Vector3d& p_c) const;

        float fx, fy, cx, cy;
        const float* w;
    };

    inline Eigen::Vector2d edge_semantic_err::cam2pixel(const Eigen::Vector3d& p_c) const {
        double x = p_c[0] / p_c[2], y = p_c[1] / p_c[2];
        return { fx * x + cx, fy * y + cy };
    }
    
} // namespace vso

#endif