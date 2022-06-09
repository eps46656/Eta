#ifndef ETA_MODEL_CUH
#define ETA_MODEL_CUH

#include "Matrix.cuh"
#include "TracingBatch.cuh"

namespace eta {

class Model {
public:
    virtual ~Model() = default;

    __host__ virtual void LoadOn(cudaStream_t stream) = 0;

    __host__ virtual void
    Rasterize(int height, int width, //

              float* dst_w, //
              Material* dst_material, //

              int* locker, //

              Matrix<float, 3, 4> camera_mat, // camera mat

              cudaStream_t stream // stream
    ) const = 0;

    __host__ virtual void RayCast(int num_of_rays, //

                                  float* dst_w, //
                                  Material* dst_material, //

                                  float* eff, //
                                  Vector<float, 3>* ray_origin, //
                                  Vector<float, 3>* ray_direct, //
                                  int* locker, //

                                  cudaStream_t stream //
    ) const = 0;
};

Matrix<float, 4, 4>
normalize_vertex_coord(const std::vector<float>& vertex_coord);

} // namespace eta

#endif
