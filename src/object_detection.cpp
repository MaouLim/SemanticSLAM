#include "object_detection.hpp"

#include <sstream>

namespace obj_slam {

    fake_detector::fake_detector(const std::string& precalc_path) {
        size_t count = 0;
        char fname[16];

        std::ifstream fin(precalc_path + "/times.txt");
        assert(fin.good());

        while (!fin.eof()) {
            std::string line;
            std::getline(fin, line);

            if (!line.empty()) {
                std::stringstream sstream(line);
                double ts;
                sstream >> ts;
                
                sprintf(fname, "/obj/%.6d.txt", count);
                _obj_map[ts] = precalc_path + std::string(fname);
                ++count;
            }
        }
    }

    std::vector<detected_bbox> 
    fake_detector::detect(const cv::Mat& rgb_img, double timestamp) {
        std::ifstream fin(_obj_map[timestamp], std::ios_base::in);
        assert(fin.good());

        std::vector<detected_bbox> res;
        while (!fin.eof()) {
            std::string line;
            std::getline(fin, line);
            if (line.empty()) { continue; }

            detected_bbox obj;
            obj.cls = 0;

            std::stringstream sstream(line);
            sstream >> obj.prob;
            for (int i = 0; i < 4; ++i) { sstream >> obj.xywh[i]; }
            res.push_back(obj);
        }

        return res;
    }
}