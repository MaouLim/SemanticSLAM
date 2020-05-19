#ifndef _SEMANTIC_CLASSIFIER_HPP_
#define _SEMANTIC_CLASSIFIER_HPP_

#include <opencv2/opencv.hpp>

namespace vso {

    struct semantic_lab;

    struct semantic_classifier {
        virtual ~semantic_classifier() = default;
        virtual semantic_lab* compute(const cv::Mat& img_color, int id) = 0;
    };
    
    struct fake_classifier : semantic_classifier {

        explicit fake_classifier(const std::string& precalc_path) : 
            _precalc_path(precalc_path) { }
        virtual ~fake_classifier() = default;

        semantic_lab* compute(const cv::Mat& img_color, int id) override;
    
    private:
        std::string _precalc_path;
    };

} // namespace vso


#endif