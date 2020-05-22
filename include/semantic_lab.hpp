#ifndef _SEMANTIC_LABEL_HPP_
#define _SEMANTIC_LABEL_HPP_

#include <opencv2/opencv.hpp>

namespace vso {

    struct semantic_lab {
        virtual ~semantic_lab() = default;
        virtual void logits(float x, float y, float* logits) = 0;
        virtual void probability_vec(float x, float y, float* p) = 0;
    };

    /**
     * @brief cityscape label information from rgb image-label
     */ 
    struct cityscape : semantic_lab {

        enum catagory {
            ROAD, SIDEWALK,      BUILDING,     WALL,       FENCE,
            POLE, TRAFFIC_LIGHT, TRAFFIC_SIGN, VEGETATION, TERRAIN,
            SKY,  PERSON,        RIDER,        CAR,        TRUCK,
            BUS,  TRAIN,         MOTOCYCLE,    BICYCLE,    NONE
        };

        static constexpr int n_channels     = 3;
        static constexpr int n_real_classes = 19;
        static constexpr int n_classes      = n_real_classes + 1;

        static const uchar rgb_index[n_classes * n_channels];

        cityscape(const cv::Mat& semantic_map, double sigma);

        static cv::Vec3b color_of(int cls_idx);
        static int catagory_of(const cv::Vec3b& rgb);
        static int catagory_of(uchar r, uchar g, uchar b);

        double sigma() const { return _sigma; }
        void set_sigma(double sigma);

        void logits(float x, float y, float* logits) override;
        void logits(int x, int y, float* logits);

        void probability_vec(float x, float y, float* p) override;
        void probability_vec(int x, int y, float* p);

        bool cache_available() const { return _cache_available; }
        void clear_cache();

    private:
        void _compute_cache();
        bool _check_uv(float u, float v, float border = 0.f) const;

        double  _sigma;
        cv::Mat _semantic_map;
    public:
        bool    _cache_available;
        cv::Mat _dist_maps[n_classes];
    };

    inline bool cityscape::_check_uv(float u, float v, float border) const {
        assert(0 <= border);
        return border <= u && int(u + border) < _semantic_map.cols &&
               border <= v && int(v + border) < _semantic_map.rows;
    }

    struct cityscape5 : semantic_lab {

        enum catagory {
                  ROAD = 0, SIDEWALK = 1, BUILDING = 2, 
            VEGETATION = 3,      CAR = 4,     NONE = 5
        };

        static constexpr int n_channels     = 3;
        static constexpr int n_real_classes = 5;
        static constexpr int n_classes      = n_real_classes + 1;

        static const uchar rgb_index[n_classes * n_channels];

        cityscape5(const cv::Mat& semantic_map, double sigma);

        static cv::Vec3b color_of(int cls_idx);
        static int catagory_of(const cv::Vec3b& rgb);
        static int catagory_of(uchar r, uchar g, uchar b);

        double sigma() const { return _sigma; }
        void set_sigma(double sigma);

        void logits(float x, float y, float* logits) override;
        void logits(int x, int y, float* logits);

        void probability_vec(float x, float y, float* p) override;
        void probability_vec(int x, int y, float* p);

        bool cache_available() const { return _cache_available; }
        void clear_cache();

    private:
        void _compute_cache();
        bool _check_uv(float u, float v, float border = 0.f) const;

        double  _sigma;
        cv::Mat _semantic_map;
    public:
        bool    _cache_available;
        cv::Mat _dist_maps[n_classes];
    };

    inline bool cityscape5::_check_uv(float u, float v, float border) const {
        assert(0 <= border);
        return border <= u && int(u + border) < _semantic_map.cols &&
               border <= v && int(v + border) < _semantic_map.rows;
    }

    /**
     * @brief cityscape label information from deep neuron network 
     *        output logits
     */
    struct cityscape_dl : semantic_lab {

    };
    
} // namespace vso


#endif