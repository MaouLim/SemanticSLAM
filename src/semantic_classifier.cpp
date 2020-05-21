#include "semantic_classifier.hpp"
#include "semantic_lab.hpp"

namespace vso {

    fake_classifier::fake_classifier(const std::string& precalc_path) {

        size_t count = 0;
        char fname[17];

        std::ifstream fin(precalc_path + "/times.txt");
        assert(fin.good());

        while (!fin.eof()) {
            std::string line;
            std::getline(fin, line);

            if (!line.empty()) {
                std::stringstream sstream(line);
                double ts;
                sstream >> ts;
                
                sprintf(fname, "/mask/%.6d.png", count);
                _mask_map[ts] = precalc_path + std::string(fname);
                ++count;
            }
        }
    }

    std::shared_ptr<semantic_lab> 
    fake_classifier::compute(const cv::Mat& img_color, double timestamp) {
        cv::Mat semantic = cv::imread(_mask_map[timestamp]);
        cv::cvtColor(semantic, semantic, cv::COLOR_BGR2RGB);
        assert(semantic.data);
        return std::make_shared<cityscape>(semantic, 10.);
    }
    
} // namespace vso
