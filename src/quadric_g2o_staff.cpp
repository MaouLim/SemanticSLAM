#include "quadric_g2o_staff.hpp"

namespace g2o {

    edge_quadric_proj::edge_quadric_proj(
        float fx, float fy, float cx, float cy, int h, int w
    ) {
        cam_mat = Eigen::Matrix3d::Identity();
        cam_mat(0, 0) = fx;
        cam_mat(1, 1) = fy;
        cam_mat(0, 2) = cx;
        cam_mat(1, 2) = cy;

        height = h;
        width  = w;
    }

    void edge_quadric_proj::computeError() {
        VertexSE3Expmap* v0 = (VertexSE3Expmap*) _vertices[0];
        vertex_quadric*  v1 = (vertex_quadric*)  _vertices[1];

        Sophus::SE3d pose = tools::to_sophus(v0->estimate());
        const obj_slam::ellipsoid& obj = v1->estimate();

        Eigen::Vector4d bbox_proj = obj.bbox_on(pose, cam_mat, height, width);
        _error = (bbox_proj - _measurement).array().square();
    }

    edge_quadric_proj_iou::edge_quadric_proj_iou(
        float fx, float fy, float cx, float cy, int h, int w
    ) {
        cam_mat = Eigen::Matrix3d::Identity();
        cam_mat(0, 0) = fx;
        cam_mat(1, 1) = fy;
        cam_mat(0, 2) = cx;
        cam_mat(1, 2) = cy;

        height = h;
        width  = w;
    }

    void edge_quadric_proj_iou::computeError() {
        VertexSE3Expmap* v0 = (VertexSE3Expmap*) _vertices[0];
        vertex_quadric*  v1 = (vertex_quadric*)  _vertices[1];

        Sophus::SE3d pose = tools::to_sophus(v0->estimate());
        const obj_slam::ellipsoid& obj = v1->estimate();

        Eigen::Vector4d bbox_proj = obj.bbox_on(pose, cam_mat, height, width);
        double iou = tools::box_iou(bbox_proj, _measurement);
        if (iou < 1e-6) { assert(0. < iou); iou = 1e-6; }
        _error(0, 0) = -log(iou);
    }

}