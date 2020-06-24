/*
 * Created by Maou Lim on 2020/6/23.
 */
#include <fstream>
#include <iostream>
#include <vector>

#include <statistics.h>
#include <Eigen/Core>

#include "common.hpp"

double _match(
	const std::vector<Eigen::Vector3d>& cloud0,
	const std::vector<Eigen::Vector3d>& cloud1
) {

	double score = 1.;
	alglib::real_1d_array arr0, arr1;
	double p = 0, p0 = 0, p1 = 0;

	for (int dim = 0; dim < 3; ++dim) {

		arr0.setlength(cloud0.size());
		for (auto i = 0; i < cloud0.size(); ++i) {
			arr0[i] = cloud0[i][dim];
		}
		arr1.setlength(cloud1.size());
		for (auto i = 0; i < cloud1.size(); ++i) {
			arr1[i] = cloud1[i][dim];
		}

		std::cout << arr0.tostring(4) << std::endl;
		std::cout << arr1.tostring(4) << std::endl;
		alglib::studentttest2(arr0, arr0.length(), arr1, arr1.length(), p, p0, p1);
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
		if (line.empty()) { continue; }
		std::stringstream stream(line);
		char dim;
		int idx;
		stream >> idx;
		if (idx != cur_idx) {
			if (-1 != cur_idx) {
				res.push_back(std::move(tmp));
			}
			cur_idx = idx;
		}
		for (int i = 0; i < 3; ++i) {
			stream >> dim;
			stream >> vec[i];
		}

		tmp.push_back(vec);
	}

	if (!tmp.empty()) {
		res.push_back(std::move(tmp));
	}

	return res;
}

int main(int argc, char** argv) {

	auto clouds2 = read_clouds("data/pc/point_cloud2.csv");
	auto clouds3 = read_clouds("data/pc/point_cloud3.csv");

	std::cout << clouds2.size() << std::endl;
	std::cout << clouds3.size() << std::endl;

	for (auto i = 0; i < clouds2.size(); ++i) {
		std::cout << "clouds2-" << i << ": " << clouds2[i].size() << std::endl;
	}
	for (auto i = 0; i < clouds3.size(); ++i) {
		std::cout << "clouds3-" << i << ": " << clouds3[i].size() << std::endl;
	}

	auto& c0 = clouds2[0];
	auto& c1 = clouds3[0];

	tools::filter_point_cloud(c0);
	tools::filter_point_cloud(c1);

	std::cout << c0.size() << std::endl;
	std::cout << c1.size() << std::endl;

	double score = _match(c0, c1);
	std::cout << score << std::endl;
	return 0;
}