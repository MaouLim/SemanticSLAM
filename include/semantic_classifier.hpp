#ifndef _SEMANTIC_CLASSIFIER_HPP_
#define _SEMANTIC_CLASSIFIER_HPP_

#include <memory>
#include <opencv2/opencv.hpp>

namespace vso {

    struct semantic_lab;

    struct semantic_classifier {
        virtual ~semantic_classifier() = default;
        virtual std::shared_ptr<semantic_lab> compute(const cv::Mat& img_color, double timestamp) = 0;
    };
    
    struct fake_classifier : semantic_classifier {

        explicit fake_classifier(const std::string& precalc_path);
        virtual ~fake_classifier() = default;

        std::shared_ptr<semantic_lab> compute(const cv::Mat& img_color, double timestamp) override;
    
    private:
        std::unordered_map<double, std::string> _mask_map;
    };

} // namespace vso


#endif