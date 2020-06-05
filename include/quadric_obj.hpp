#ifndef _QUADRIC_OBJECT_HPP_
#define _QUADRIC_OBJECT_HPP_

#include "ellipsoid.hpp"
#include "obj_det.hpp"

namespace obj_slam {

    struct quadric_frame;

    struct quadric_obj {

        enum state_t { UPDATING, NEED_OBSERVATIONS };

        state_t                          state;
        size_t                           id;
        ellipsoid                        param;
        std::map<quadric_frame*, size_t> observations;

        static size_t _seq_id;

        quadric_obj();
        void observe(quadric_frame* frame, size_t box_idx);
        bool available() const { return UPDATING == state; }

    private:
        ellipsoid _from_observations() const;
    };
}

#endif