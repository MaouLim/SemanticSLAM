#include "semantic_lab.hpp"
#include "semantic_classifier.hpp"

int argmax(float* p) {
    int idx = 0;
    for (int i = 0; i < vso::cityscape::n_real_classes; ++i) {
        if (p[idx] < p[i]) {
            idx = i;
        }
    }
    return idx;
}

int main(int argc, char** argv) {

    vso::fake_classifier classifier("/media/maou/TOSHIBA-Maou/KITTI/sequence/00/image_2_seg_colored/");
    auto* lab = classifier.compute(cv::Mat(), 0);

    int h = 376, w = 1241;
    cv::Mat semantic_img = cv::Mat::zeros(h, w, CV_8UC3);
    cv::Mat car_prob_map = cv::Mat::zeros(h, w, CV_32FC1);

    float p[vso::cityscape::n_classes];
    for (int r = 1; r < h - 1; ++r) {
        for (int c = 1; c < w - 1; ++c) {
            lab->probability_vec((float) c, (float) r, p);
            int cls = argmax(p);
            semantic_img.at<cv::Vec3b>(r, c) = vso::cityscape::color_of(cls);
            car_prob_map.at<float>(r, c) = p[vso::cityscape::CAR];
        }
    }
    cv::cvtColor(semantic_img, semantic_img, cv::COLOR_BGR2RGB);
    cv::imshow("recover semantic", semantic_img);
    cv::imshow("Car map", car_prob_map);

    vso::cityscape* cs_lab = (vso::cityscape*) lab;
    cv::imshow("Car dist map", cs_lab->_dist_maps[vso::cityscape::CAR]);
    

    cv::waitKey();
    
    delete lab;
    return 0;
}