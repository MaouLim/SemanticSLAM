#ifndef _COMMON_HPP_
#define _COMMON_HPP_

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus_templ/se3.hpp>
#include <g2o/types/types_six_dof_expmap.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>

namespace tools {

	inline bool solve_quadratic_equation(
		double a, double b, double c, double& x0, double& x1
	) {
		assert(0. != a);
		double delta = b * b - 4 * a * c;
		if (delta < 0.) { return false; }
		double delta_sqrt = std::sqrt(delta);
		x0 = (-b + delta_sqrt) / (2. * a);
		x1 = (-b - delta_sqrt) / (2. * a);
		return true;
	}

	inline Sophus::SE3d to_sophus_se3(const cv::Mat& pose) {
		Eigen::Matrix<double,3,3> R;
		R << pose.at<float>(0,0), pose.at<float>(0,1), pose.at<float>(0,2),
			pose.at<float>(1,0), pose.at<float>(1,1), pose.at<float>(1,2),
			pose.at<float>(2,0), pose.at<float>(2,1), pose.at<float>(2,2);
		Eigen::Quaterniond q(R);
		Eigen::Vector3d t(pose.at<float>(0,3), pose.at<float>(1,3), pose.at<float>(2,3));
		return Sophus::SE3d(q, t);
	}
}

namespace tools {

	inline Sophus::SE3d to_sophus(const g2o::SE3Quat& pose) {
		return Sophus::SE3d(pose.rotation(), pose.translation());
	}

	/**
	 * @brief calculate the overlap of the [a0, a1] with [b0, b1]
	 */
	inline double overlap(double a0, double a1, double b0, double b1) {
		double left  = a0 > b0 ? a0 : b0;
		double right = a1 < b1 ? a1 : b1;
		return right - left;
	}

	/**
	 * @brief calculate the box_intersection of two boxes
	 * @note box represented as (xmin, ymin, xmax, ymax)
	 */
	inline double box_intersection(
		const Eigen::Vector4d& box0,
		const Eigen::Vector4d& box1
	) {
		double w = overlap(box0[0], box0[2], box1[0], box1[2]);
		double h = overlap(box0[1], box0[3], box1[1], box1[3]);
		return w * h;
	}

	inline double box_area(const Eigen::Vector4d& box) {
		return (box[2] - box[0]) * (box[3] - box[1]);
	}

	inline double box_iou(
		const Eigen::Vector4d& box0,
		const Eigen::Vector4d& box1
	) {
		double i = box_intersection(box0, box1);
		double u = box_area(box0) + box_area(box1) - i;
		return i / u;
	}
}

namespace tools {

	inline pcl::PointXYZ to_pcl(const Eigen::Vector3d& p) {
		return pcl::PointXYZ(p[0], p[1], p[2]);
	}

	inline Eigen::Vector3d to_eigen(const pcl::PointXYZ& p) {
		return Eigen::Vector3d(p.x, p.y, p.z);
	}

	inline void filter_point_cloud(std::vector<Eigen::Vector3d>& cloud) {
		if (cloud.size() < 20) { return; }

		pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto& each : cloud) {
			pcl_cloud->push_back(to_pcl(each));
		}

		int k = pcl_cloud->size() / 3;
		if (k < 20) { k = 20; }

		std::vector<int> indices;
		pcl::StatisticalOutlierRemoval<pcl::PointXYZ> filter;
		filter.setInputCloud(pcl_cloud);
		filter.setMeanK(k);
		filter.setStddevMulThresh(0.8);
		filter.filter(indices);

		cloud.clear();
		for (auto idx : indices) {
			cloud.push_back(to_eigen(pcl_cloud->points[idx]));
		}
	}
}

#endif
