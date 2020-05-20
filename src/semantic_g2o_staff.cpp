#include "semantic_g2o_staff.hpp"
#include "semantic_lab.hpp"

namespace vso {

    void edge_semantic_err::computeError() {
        g2o::VertexSE3Expmap*   v0 = (g2o::VertexSE3Expmap*)  _vertices[0];
        g2o::VertexSBAPointXYZ* v1 = (g2o::VertexSBAPointXYZ) _vertices[1];
        Eigen::Vector2d uv = cam2pixel(v0->estimate().map(v1->estimate()));
        float p[cityscape::n_classes];
        _measurement->logits((float) uv.x(), (float) uv.y(), p);
        _error.setZero();
        for (int i = 0; i < cityscape::n_classes; ++i) {
            _error(0) += w[i] * p[i];
        }
    }
}



