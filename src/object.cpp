#include "object.hpp"

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
		// todo: point_cloud init
		this->add_observation(first_ob);
	}

	object::~object() {
		for (auto& ptr : observations) {
			delete ptr;
			ptr = nullptr;
		}
	}

	object::object(object&& rhs) noexcept :
		id(rhs.id), history_centers(std::move(rhs.history_centers)),
		observations(std::move(rhs.observations)),// todo: point_cloud init
		n_points(rhs.n_points), quadric_built(false) { }

	object& object::operator=(object&& rhs) noexcept {
		if (this == &rhs) { return *this; }

		id = rhs.id;
		history_centers = std::move(rhs.history_centers);
		observations = std::move(rhs.observations);
		n_points = rhs.n_points;
		param = rhs.param;
		quadric_built = rhs.quadric_built;
		// todo: point_cloud copy
		return *this;
	}

	void object::add_observation(obj_observation* ob) {
		this->observations.push_back(ob);
		this->n_points += ob->point_cloud.size();
		this->_sampling_point_cloud(/* todo size */100);
		this->_compute_point_cloud_center();
	}

	void object::merge(object&& obj) {
		std::copy(obj.history_centers.begin(), obj.history_centers.end(), history_centers.end());
		std::copy(obj.observations.begin(), obj.observations.end(), observations.end());
		n_points += obj.n_points;

		obj.history_centers.clear();
		obj.observations.clear();
		obj.n_points = 0;

		// todo merge point_cloud
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
			this->_merge_ob(associated_idx, ob);
		}
	}

	void object_manager::optimize() {

	}

	int object_manager::_find_association(const obj_observation* ob) {

	}

	void object_manager::_merge_ob(int obj_idx, obj_observation* ob) {
		objects[obj_idx]->add_observation(ob);
	}

	void object_manager::_create_new_obj(obj_observation* ob) {
		objects.push_back(new object(ob));
	}
}