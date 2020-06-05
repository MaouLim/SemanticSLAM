#ifndef _QUADRIC_FRAME_HPP_
#define _QUADRIC_FRAME_HPP_

#include "obj_det.hpp"

#include <opencv2/opencv.hpp>
#include <sophus_templ/se3.hpp>

namespace g2o { class VertexSE3Expmap; }

namespace obj_slam {

    struct quadric_frame {

        size_t                     id;
        cv::Mat                    raw_image;
        Sophus::SE3d               t_cw;
        Sophus::SE3d               t_wc;
        Eigen::Matrix3d            cam_mat;
        std::vector<detected_bbox> bboxes;

        g2o::VertexSE3Expmap* v;
        

    };
}

#endif