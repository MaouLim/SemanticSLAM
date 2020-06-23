#ifndef _OBJECT_HPP_
#define _OBJECT_HPP_

#include <mutex>

#include <Eigen/Core>
#include <sophus_templ/se3.hpp>

#include "object_detection.hpp"
#include "ellipsoid.hpp"

namespace ORB_SLAM2 { class KeyFrame; }

namespace obj_slam {

	struct obj_observation {

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		ORB_SLAM2::KeyFrame*         key_frame;
		Sophus::SE3d                 t_cw;
		detected_bbox                bbox;
		std::vector<Eigen::Vector3d> point_cloud;

		obj_observation() : key_frame(nullptr) { }

		obj_observation(const obj_observation&) = delete;
		obj_observation& operator=(const obj_observation&) = delete;

		obj_observation(obj_observation&&) noexcept;
		obj_observation& operator=(obj_observation&&) noexcept;
	};

    struct object {

	    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    	static constexpr int max_feature_points = 100;

	    size_t                        id;
        std::vector<Eigen::Vector3d>  history_centers;
        std::vector<Eigen::Vector3d>  point_cloud;
		std::vector<obj_observation*> observations;
		size_t                        n_points;

		bool                          quadric_built;
	    ellipsoid                     param;

		object() : id(_id_seq++), n_points(0), quadric_built(false) { }
		explicit object(obj_observation* first_ob);
		~object();
		object(const object&) = delete;
		object(object&& rhs) noexcept;

		object& operator=(const object&) = delete;
		object& operator=(object&& rhs) noexcept;

		bool build_quadric(const Eigen::Matrix3d& cam_mat);
		void add_observation(obj_observation* ob);
		void merge(object&& obj);

    private:
    	void _compute_point_cloud_center();
    	void _sampling_point_cloud(int max_pts = max_feature_points);

    	static std::vector<Eigen::Vector3d> _fuse(
    		const std::vector<Eigen::Vector3d>& pc0,
    		const std::vector<Eigen::Vector3d>& pc1,
		    int                                 max_pts
    	);

		static size_t _id_seq;
    };

    struct object_manager {

    	std::mutex           obj_mtx;
	    std::vector<object*> objects;

	    object_manager() = default;
	    ~object_manager();

		void handle_observation(obj_observation* ob);
		void optimize();

    private:
	    int _find_association(const obj_observation* ob);
	    void _merge_ob(int obj_idx, obj_observation* ob);
	    void _create_new_obj(obj_observation* ob);

	    static double _match(
	    	const std::vector<Eigen::Vector3d>& cloud0,
	    	const std::vector<Eigen::Vector3d>& cloud1
	    );
    };
}

#endif