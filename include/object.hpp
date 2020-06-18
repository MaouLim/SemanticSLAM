#ifndef _OBJECT_HPP_
#define _OBJECT_HPP_

#include <list>
#include <map>
#include <set>
#include <Eigen/Core>

namespace ORB_SLAM2 {  

    class Frame;
    class KeyFrame;
    class MapPoint;
}

namespace obj_slam {

    struct object {

        std::set<ORB_SLAM2::MapPoint*>      map_points;
        std::map<ORB_SLAM2::Frame*, size_t> observations;
        std::vector<Eigen::Vector3d>        history_centers;

        
    };
}

#endif