#include "semantic_classifier.hpp"
#include "semantic_lab.hpp"

namespace vso {

    semantic_lab* fake_classifier::compute(const cv::Mat& img_color, int id) {
        char fname_chs[11];
        sprintf(fname_chs, "%.6d.png", id);
        std::string fname(fname_chs);
        cv::Mat semantic = cv::imread(_precalc_path + "/" + fname);
        cv::cvtColor(semantic, semantic, cv::COLOR_BGR2RGB);
        assert(semantic.data);
        return new cityscape(semantic, 10.);
    }
    
} // namespace vso
