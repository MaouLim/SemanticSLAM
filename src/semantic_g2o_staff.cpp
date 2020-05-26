#include "semantic_g2o_staff.hpp"
#include "semantic_lab.hpp"

namespace g2o {

    void edge_semantic_err::computeError() {

	    if (!_measurement || w.size() != vso::cityscape5::n_classes) {
	    	this->setLevel(1); return;
	    }

        g2o::VertexSE3Expmap*   v0 = (g2o::VertexSE3Expmap*)   _vertices[0];
        g2o::VertexSBAPointXYZ* v1 = (g2o::VertexSBAPointXYZ*) _vertices[1];

        Eigen::Vector2d uv = cam2pixel(v0->estimate().map(v1->estimate()));

        float p[vso::cityscape5::n_classes];
        auto ret = _measurement->logits((float) uv.x(), (float) uv.y(), p);
        if (!ret) { return; }

        _error.setZero();
        for (int i = 0; i < vso::cityscape5::n_classes; ++i) {
            _error(0) += w[i] * p[i];
        }
        _error *= lamda;
    }

    bool edge_semantic_err::in_viewport(
        const Eigen::Vector2d& uv, double border
    ) const {
        return border <= uv.x() && int(border + uv.x()) < width && 
               border <= uv.y() && int(border + uv.y()) < height;
    }
}



