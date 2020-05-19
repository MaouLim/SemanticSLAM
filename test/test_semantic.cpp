#include "semantic_lab.hpp"
#include "semantic_classifier.hpp"

int argmax(float* p) {
    int idx = 0;
    for (int i = 0; i < 20; ++i) {
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

    float p[20];
    for (int r = 1; r < h - 1; ++r) {
        for (int c = 1; c < w - 1; ++c) {
            lab->probability_vec(c, r, p);
            int cls = argmax(p);
            semantic_img.at<cv::Vec3b>(r, c) = vso::cityscape::color_of(cls);
        }
    }
    cv::cvtColor(semantic_img, semantic_img, cv::COLOR_BGR2RGB);
    cv::imshow("recover semantic", semantic_img);
    cv::waitKey();
    
    delete lab;
    return 0;
}