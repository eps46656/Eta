#ifndef ETA_RAYCAST_CUH
#define ETA_RAYCAST_CUH

#include "Vector.cuh"

namespace eta {

template<typename MaterialGetter>
__global__ void RayCast(int num_of_rays, //

                        float* dst_w, //
                        Material* dst_material, //

                        float* eff, //
                        Vector<float, 3>* ray_origin, //
                        Vector<float, 3>* ray_direct, //

                        int num_of_faces, //
                        View face_coord, // Vector<float, 3> [num_of_faces, 3]

                        Matrix<float, 4, 4> transform, //

                        MaterialGetter material_getter //
);

} // namespace eta

#endif
