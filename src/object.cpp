#include "object.hpp"

#include <statistics.h>

#include "Frame.h"

namespace obj_slam {

	obj_observation::obj_observation(obj_observation&& rhs) noexcept :
		key_frame(rhs.key_frame), t_cw(rhs.t_cw), bbox(rhs.bbox),
		point_cloud(std::move(rhs.point_cloud)) { }

	obj_observation& obj_observation::operator=(obj_observation&& rhs) noexcept {
		if (this == &rhs) { return *this; }

		key_frame = rhs.key_frame;
		t_cw = rhs.t_cw;
		bbox = rhs.bbox;
		point_cloud = std::move(point_cloud);
		return *this;
	}

	size_t object::_id_seq = 0;

	object::object(obj_observation* first_ob) :
		id(_id_seq++), n_points(0), quadric_built(false)
	{
		this->add_observation(first_ob);
		this->_sampling_point_cloud();
	}

	object::~object() {
		for (auto& ptr : observations) {
			delete ptr;
			ptr = nullptr;
		}
	}

	object::object(object&& rhs) noexcept :
		id(rhs.id), history_centers(std::move(rhs.history_centers)),
		point_cloud(std::move(rhs.point_cloud)),
		observations(std::move(rhs.observations)),
		n_points(rhs.n_points), quadric_built(false) { }

	object& object::operator=(object&& rhs) noexcept {
		if (this == &rhs) { return *this; }

		id = rhs.id;
		history_centers = std::move(rhs.history_centers);
		point_cloud = std::move(rhs.point_cloud);
		observations = std::move(rhs.observations);
		n_points = rhs.n_points;
		param = rhs.param;
		quadric_built = rhs.quadric_built;

		return *this;
	}

	void object::add_observation(obj_observation* ob) {
		this->observations.push_back(ob);
		this->n_points += ob->point_cloud.size();
		this->_sampling_point_cloud();
		this->_compute_point_cloud_center();
	}

	void object::merge(object&& obj) {
		std::copy(obj.history_centers.begin(), obj.history_centers.end(), history_centers.end());
		std::copy(obj.observations.begin(), obj.observations.end(), observations.end());
		n_points += obj.n_points;
		point_cloud = _fuse(point_cloud, obj.point_cloud, max_feature_points);

		obj.history_centers.clear();
		obj.point_cloud.clear();
		obj.observations.clear();
		obj.n_points = 0;
	}

	bool object::build_quadric(const Eigen::Matrix3d& cam_mat) {
		if (observations.size() < 3) { return false; }

		std::vector<Eigen::Vector4d> planes;
		planes.reserve(observations.size() * 4);

		for (auto& ob : observations) {

			const int* xywh = ob->bbox.xywh;
			const int bbox[4] = { xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3] };

			// plane = (KT)' * line
			Eigen::Matrix<double, 4, 3> KT_t =
				(cam_mat * ob->t_cw.matrix3x4()).transpose();

			planes.emplace_back(KT_t * Eigen::Vector3d(1, 0, -bbox[0]));
			planes.emplace_back(KT_t * Eigen::Vector3d(0, 1, -bbox[1]));
			planes.emplace_back(KT_t * Eigen::Vector3d(1, 0, -bbox[2]));
			planes.emplace_back(KT_t * Eigen::Vector3d(0, 1, -bbox[3]));
		}

		Eigen::MatrixXd A(planes.size(), 10);
		for (size_t i = 0; i < planes.size(); ++i) {
			const auto& plane = planes[i];

			double pi1 = plane[0];
			double pi2 = plane[1];
			double pi3 = plane[2];
			double pi4 = plane[3];

			// Aj
			A(i, 0) = pi1 * pi1;
			A(i, 1) = 2. * pi1 * pi2;
			A(i, 2) = 2. * pi1 * pi3;
			A(i, 3) = 2. * pi1 * pi4;
			A(i, 4) = pi2 * pi2;
			A(i, 5) = 2. * pi2 * pi3;
			A(i, 6) = 2. * pi2 * pi4;
			A(i, 7) = pi3 * pi3;
			A(i, 8) = 2. * pi3 * pi4;
			A(i, 9) = pi4 * pi4;
		}

		/**
		 * @note A least squares solution q(minimal of Q*) that minimizes ||Aq||
		 *       can be obtained as the last colunm of V, which Aq = UDV'
		 * @cite QUADRICSLAM: DUAL QUADRICS FROM OBJECT DETECTIONS AS LANDMARKS
		 */
		Eigen::JacobiSVD<Eigen::MatrixXd> solver(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Vector10d q = solver.matrixV().col(9);
		param = ellipsoid(q);

		quadric_built = true;
		return true;
	}

	void object::_compute_point_cloud_center() {
		Eigen::Vector3d center(0, 0, 0);

		for (auto& pt : point_cloud) {
			center += pt;
		}

		center /= (double) point_cloud.size();
		history_centers.push_back(center);
	}

	void object::_sampling_point_cloud(int max_pts) {
		// simply sampling the latest observation point cloud
		point_cloud = _fuse(point_cloud, observations.back()->point_cloud, max_pts);
	}

	std::vector<Eigen::Vector3d> object::_fuse(
		const std::vector<Eigen::Vector3d>& pc0,
		const std::vector<Eigen::Vector3d>& pc1,
		int                                 max_pts
	) {
		size_t total_sz = pc0.size() + pc1.size();
		std::cout << "total_sz: " << total_sz << std::endl;
		std::vector<int> indices(total_sz);
		for (size_t i = 0; i < total_sz; ++i) {
			indices[i] = i;
		}

		std::vector<Eigen::Vector3d> tmp;
		tmp.reserve(max_pts);
		std::shuffle(indices.begin(), indices.end(), std::default_random_engine());
		for (auto i = 0; i < max_pts; ++i) {
			int idx = indices[i];
			if (pc0.size() <= idx) {
				idx -= pc0.size();
				tmp.push_back(pc1[idx]);
			}
			else { tmp.push_back(pc0[idx]); }
		}

		return tmp;
	}

	object_manager::~object_manager() {
		for (auto& ptr : objects) {
			if (ptr) {
				delete ptr;
				ptr = nullptr;
			}
		}
		objects.clear();
	}

	void object_manager::handle_observation(obj_observation* ob) {
		std::lock_guard<std::mutex> lock(obj_mtx);
		int associated_idx = this->_find_association(ob);
		if (associated_idx < 0) {
			this->_create_new_obj(ob);
		}
		else {
			std::cout << "Association found: " << associated_idx << std::endl;
			this->_merge_ob(associated_idx, ob);
		}
	}

	void object_manager::optimize() {
		// todo
	}

	int object_manager::_find_association(const obj_observation* ob) {
		for (auto& obj : objects) {

		}

		return 0;
	}

	void object_manager::_merge_ob(int obj_idx, obj_observation* ob) {
		objects[obj_idx]->add_observation(ob);
	}

	void object_manager::_create_new_obj(obj_observation* ob) {
		objects.push_back(new object(ob));
	}

	double object_manager::_match(
		const std::vector<Eigen::Vector3d>& cloud0,
		const std::vector<Eigen::Vector3d>& cloud1
	) {
		alglib::real_1d_array arr0, arr1;
		double score = 1.;
		double p, p0, p1;

		for (int dim = 0; dim < 3; ++dim) {

			arr0.setlength(cloud0.size());
			for (auto i = 0; i < cloud0.size(); ++i) {
				arr0[i] = cloud0[i][dim];
			}
			arr1.setlength(cloud1.size());
			for (auto i = 0; i < cloud1.size(); ++i) {
				arr1[i] = cloud1[i][dim];
			}

			alglib::mannwhitneyutest(arr0, arr0.length(), arr1, arr1.length(), p, p0, p1);
			if (p < 0.05) { return 0.; }
			score *= p;
		}

		return score;
	}
}