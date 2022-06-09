#ifndef ETA_POINTLIGHT_CUH
#define ETA_POINTLIGHT_CUH

#include "Light.cuh"
#include "Buffer.cuh"
#include "TextureCube.cuh"
#include "Model.cuh"
#include "TracingBatch.cuh"

namespace eta {

class PointLight: public Light {
public:
    Vector<float, 3> origin;
    Vector<float, 3> intensity;

    __host__ __device__ const TextureCube<int>& cage() const;
    __host__ __device__ int cage_resolution() const;

    __host__ void InitCage(int resolution, cudaStream_t stream);

    __host__ void GenerateCage(const Model* model, cudaStream_t stream);

    __host__ void LoadOn(cudaStream_t stream);

    __host__ void Shade(int size, Vector<int, 3>* pixel,
                        TracingBatch tracing_batch,
                        cudaStream_t stream) const override;

private:
    int cage_resolution_{ 0 };
    TextureCube<float> cage_;

    Buffer<GPU> cage_w_[6];
};

} // namespace eta

#endif
