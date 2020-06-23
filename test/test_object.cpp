/*
 * Created by Maou Lim on 2020/6/23.
 */
#include <fstream>
#include <iostream>
#include <vector>

#include <statistics.h>
#include <Eigen/Core>

double _match(
	const std::vector<Eigen::Vector3d>& cloud0,
	const std::vector<Eigen::Vector3d>& cloud1
) {

	double score = 1.;
	double p, p0, p1;

	for (int dim = 0; dim < 3; ++dim) {

		alglib::real_1d_array arr0, arr1;

		arr0.setlength(cloud0.size());
		for (auto i = 0; i < cloud0.size(); ++i) {
			arr0[i] = cloud0[i][dim];
		}
		arr1.setlength(cloud1.size());
		for (auto i = 0; i < cloud1.size(); ++i) {
			arr1[i] = cloud1[i][dim];
		}

		alglib::mannwhitneyutest(arr0, arr0.length(), arr1, arr1.length(), p, p0, p1);
		std::cout << "p: " << p << std::endl;
		//if (p < 0.05) { return 0.; }
		score *= p;
	}

	return score;
}

std::vector<std::vector<Eigen::Vector3d>> read_clouds(const std::string& path) {
	std::ifstream fin(path);
	assert(fin.good());

	std::vector<std::vector<Eigen::Vector3d>> res;

	std::vector<Eigen::Vector3d> tmp;
	int cur_idx = -1;
	Eigen::Vector3d vec;

	while (!fin.eof()) {
		std::string line;
		std::getline(fin, line);
		std::stringstream stream(line);
		int idx;
		stream >> idx;
		if (idx != cur_idx) {
			if (-1 != cur_idx) {
				res.push_back(std::move(tmp));
			}
			cur_idx = idx;
		}

		stream >> vec[0] >> vec[1] >> vec[2];
		tmp.push_back(vec);
	}

	if (!tmp.empty()) {
		res.push_back(std::move(tmp));
	}

	return res;
}

int main(int argc, char** argv) {

	auto clouds2 = read_clouds("point_cloud2.txt");
	auto clouds3 = read_clouds("point_cloud3.txt");

	std::cout << clouds2.size() << std::endl;
	std::cout << clouds3.size() << std::endl;

	std::cout << clouds2[1].size() << std::endl;
	std::cout << clouds3[1].size() << std::endl;

	double score = _match(clouds2[1], clouds3[1]);
	std::cout << score << std::endl;
	return 0;
}