#ifndef ETA_RASTERIZE_CUH
#define ETA_RASTERIZE_CUH

#include "define.cuh"
#include "utils.cuh"
#include "View.cuh"

namespace eta {

template<typename MaterialGetter>
__global__ void Rasterize(int height, int width, //

                          float* dst_w, //
                          Material* dst_material, //

                          int* locker, //

                          int num_of_faces, //
                          View face_coord, // Vector<float, 3> [num_of_faces, 3]
                          Matrix<float, 3, 4> transform, //

                          MaterialGetter material_getter //
);

__global__ void RasterizationNormalize(int height, int width, //

                                       float* w, //

                                       Vector<float, 3> r, // right
                                       Vector<float, 3> u, // up
                                       Vector<float, 3> f // front
);

} // namespace eta

#endif
