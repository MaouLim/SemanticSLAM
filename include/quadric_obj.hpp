#ifndef _QUADRIC_OBJECT_HPP_
#define _QUADRIC_OBJECT_HPP_

#include "ellipsoid.hpp"
#include "obj_det.hpp"

namespace obj_slam {

    struct quadric_obj {

        size_t    id;
        ellipsoid rep;
        std::vector<detected_obj> observation;


        static size_t _seq_id;
    };

}

#endif