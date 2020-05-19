#include "semantic_lab.hpp"

namespace vso {

    static constexpr int _map_table[67] = 
    {
        19, 19, 19, 14, 19, 
        19, 19, 19, 13,  2, 
        19, 19, 19, 19, 19, 
        19,  8, 19, 19,  9, 
        19, 19, 19, 19, 19, 
         3, 15,  4, 18, 17, 
        19, 19, 11, 19, 19, 
        19, 19, 19,  7, 19, 
        19, 19,  1, 19, 19, 
        10, 16, 19,  6, 19, 
        19, 19,  0, 19, 12, 
        19, 19,  5, 19, 19, 
        19, 19, 19, 19, 19, 
        19, 19
    };

    constexpr uchar cityscape::rgb_index[n_classes * n_channels] = 
    { 
        128,  64, 128, // road 
        244,  35, 232, // sidewalk 
         70,  70,  70, // building 
        102, 102, 156, // wall 
        190, 153, 153, // fence 
        153, 153, 153, // pole 
        250, 170,  30, // traffic light 
        220, 220,   0, // traffic sign 
        107, 142,  35, // vegetation 
        152, 251, 152, // terrain 
         70, 130, 180, // sky 
        220,  20,  60, // person 
        255,   0,   0, // rider
          0,   0, 142, // car
          0,   0,  70, // truck 
          0,  60, 100, // bus 
          0,  80, 100, // train 
          0,   0, 230, // motorcycle 
        119,  11,  32, // bicycle 
          0,   0,   0  // none 
    };

    cityscape::cityscape(const cv::Mat& semantic_map, double sigma) {
        assert(CV_8UC3 == semantic_map.type() && semantic_map.data);
        _sigma = sigma;
        _semantic_map = semantic_map.clone();
        _compute_prob_maps();
    }

    cv::Vec3b cityscape::color_of(int cls_idx) {
        assert(cls_idx < n_classes);
        const uchar* p = rgb_index + cls_idx * n_channels;
        return { *p, *(p + 1), *(p + 2) };
    }

    int cityscape::catagory_of(const cv::Vec3b& rgb) {
        return catagory_of(rgb[0], rgb[1], rgb[2]);
    }

    int cityscape::catagory_of(uchar r, uchar g, uchar b) {
        return _map_table[((int) r + (int) g + (int) b) % 67];
    }

    void cityscape::set_sigma(double sigma) {
        if (sigma == _sigma) { return; }
        _sigma = sigma;
        _compute_prob_maps();
    }

    void cityscape::probability_vec(float x, float y, float* p) const {
        assert(_check_uv(x, y, 1.f));

        int ix = x, iy = y;
        float dx = x - ix, dy = y - iy;

        float w00 = (1.f - dx) * (1.f - dy);
        float w01 =         dx * (1.f - dy);
        float w10 = (1.f - dx) * dy;
        float w11 =         dx * dy;

        float sum_square = 0.f;
        float* q = p;
        for (int i = 0; i < n_classes; ++i) {
            const cv::Mat& prob_map = _prob_maps[i];
            float prob = w00 * prob_map.at<float>(iy, ix) + 
                         w01 * prob_map.at<float>(iy, ix + 1) + 
                         w10 * prob_map.at<float>(iy + 1, ix) + 
                         w11 * prob_map.at<float>(iy + 1, ix + 1);
            *q = prob * prob;
            sum_square += *q;
            ++q;
        }

        for (int i = 0; i < n_classes; ++i) {
            *p /= sum_square; ++p;
        }
    }

    void cityscape::_compute_prob_maps() {
        for (int i = 0; i < n_classes; ++i) {
            _prob_maps[i] = cv::Mat::zeros(_semantic_map.size(), CV_8UC1);
        }

        for (int r = 0; r < _semantic_map.rows; ++r) {
            for (int c = 0; c < _semantic_map.cols; ++c) {
                cv::Vec3b rgb = _semantic_map.at<cv::Vec3b>(r, c);
                int cls = catagory_of(rgb);
                _prob_maps[cls].at<uchar>(r, c) = 255;
            }
        }

        for (int i = 0; i < n_classes; ++i) {
            cv::Mat tmp;
            cv::GaussianBlur(_prob_maps[i], tmp, cv::Size2i(_sigma, _sigma), 0.);
            cv::distanceTransform(tmp, _prob_maps[i], CV_DIST_L2, 3, CV_32F);
        }
    }
    
} // namespace vso
