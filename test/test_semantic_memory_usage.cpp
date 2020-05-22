#include <chrono>

#include "semantic_lab.hpp"
#include "semantic_classifier.hpp"

namespace chrono = std::chrono;

int main(int argc, char** argv) {

    vso::fake_classifier classifier("data/");

    std::vector<std::shared_ptr<vso::semantic_lab>> labs;

    std::ifstream fin("data/times.txt");
    assert(fin.good());

    size_t ms = 0;
    size_t max_loads = 200;

    while (!fin.eof()) {
        std::string line;
        std::getline(fin, line);

        if (!line.empty()) {
            std::stringstream sstream(line);
            double ts;
            sstream >> ts;
            chrono::steady_clock::time_point start = chrono::steady_clock::now();
            auto lab = classifier.compute(cv::Mat(), ts);
            chrono::steady_clock::time_point end = chrono::steady_clock::now();
            
            labs.push_back(lab);
            auto t = chrono::duration_cast<chrono::milliseconds>(end - start).count();
            ms += t;
            if (max_loads <= labs.size()) { break; }
        }
    }

    double avg_ms = double(ms) / max_loads;
    std::cout << "compute lab avg time used: " << avg_ms << std::endl;

    ms = 0;
    for (auto& each : labs) {
        float p[6];
        chrono::steady_clock::time_point start = chrono::steady_clock::now();
        each->probability_vec(100.4, 200.7554, p);
        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        ms += t;
    }

    avg_ms = double(ms) / max_loads;
    std::cout << "first compute prob vec avg time used: " << avg_ms << std::endl;

    ms = 0;
    for (auto& each : labs) {
        float p[6];
        chrono::steady_clock::time_point start = chrono::steady_clock::now();
        each->probability_vec(100.4, 200.7554, p);
        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        ms += t;
    }

    avg_ms = double(ms) / max_loads;
    std::cout << "second compute prob vec avg time used: " << avg_ms << std::endl;
    
    return 0;
}