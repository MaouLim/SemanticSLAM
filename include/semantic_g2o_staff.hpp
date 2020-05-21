#ifndef _SEMANTIC_G2O_STAFF_HPP_
#define _SEMANTIC_G2O_STAFF_HPP_

#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

namespace vso { struct semantic_lab; }

namespace g2o {

    struct edge_semantic_err : 
        g2o::BaseBinaryEdge<
            1, vso::semantic_lab*, g2o::VertexSE3Expmap, g2o::VertexSBAPointXYZ
        > 
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        edge_semantic_err(int _height, int _width, const std::vector<float>& _w) :
            height(_height), width(_width),
            fx(0), fy(0), cx(0), cy(0), w(_w), lamda(1.f) { } 

        bool read(std::istream& is) override { return true; }
        bool write(std::ostream& os) const override { return true; }

        void computeError() override;

        Eigen::Vector2d cam2pixel(const Eigen::Vector3d& p_c) const;
        bool in_viewport(const Eigen::Vector2d& uv, double border = 1.) const;

        int height, width;
        float fx, fy, cx, cy;
        std::vector<float> w;
        float lamda;
    };

    inline Eigen::Vector2d edge_semantic_err::cam2pixel(const Eigen::Vector3d& p_c) const {
        double x = p_c[0] / p_c[2], y = p_c[1] / p_c[2];
        return { fx * x + cx, fy * y + cy };
    }
    
} // namespace g2o

#endif