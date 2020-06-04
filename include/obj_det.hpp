#ifndef _OBJECT_DETECTION_HPP_
#define _OBJECT_DETECTION_HPP_

#include <vector>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace obj_slam {

    struct detected_obj {

        Eigen::Vector4d bbox; // (xmin, ymin, xmax, ymax)
        float           prob; // prob
        int             cls;  // class_idx
    };

    struct obj_detector {

        virtual ~obj_detector() = default;
        virtual std::vector<detected_obj> detect(const cv::Mat& rgb_img, double timestamp) = 0;
    };

    struct fake_detector : obj_detector {

        explicit fake_detector(const std::string& precalc_path);
        virtual ~fake_detector() = default;

        std::vector<detected_obj> detect(const cv::Mat& rgb_img, double timestamp) override;
    
    private:
        std::unordered_map<double, std::string> _obj_map;
    };
}


#endif